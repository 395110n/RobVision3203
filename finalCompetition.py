import numpy as np
import cv2
from vis_nav_game import Player, Action, Phase
import pygame
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from multiprocessing import Pool
import time
import torch

# Set floating-point precision for matrix multiplication
torch.set_float32_matmul_precision("medium")

# Check if CUDA (GPU) is available and use it, otherwise default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize feature extractor (SIFT or ORB)
extractor = cv2.SIFT_create()  # or cv2.ORB_create()

# Define constants for different stages in visual odometry
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500  # Minimum number of features required for reliable processing

def vo_initialization(cam_width, cam_height, cam_fx, cam_fy, cam_cx, cam_cy, cam_k1, cam_k2, cam_k3, cam_p1, cam_p2):
    """
    Initialize the state of the visual odometry system.
    Sets up camera parameters and initializes variables for tracking and matching features.
    """
    vo_state = {
        'frame_stage': 0,
        'cam': {
            'width': cam_width,
            'height': cam_height,
            'fx': cam_fx,
            'fy': cam_fy,
            'cx': cam_cx,
            'cy': cam_cy,
            'k1': cam_k1,
            'k2': cam_k2,
            'k3': cam_k3,
            'p1': cam_p1,
            'p2': cam_p2
        },
        'new_frame': None,
        'last_frame': None,
        'cur_R': None,
        'cur_t': None,
        'cur_Normal': None,
        'px_ref': None,
        'px_cur': None,
        'focal': cam_fx,
        'pp': (cam_cx, cam_cy),
        'trueX': 0,
        'trueY': 0,
        'trueZ': 0,
        'detector': cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True),
        'prev_normal': None,
        'K': np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]),
        'P': np.concatenate((np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]]), np.zeros((3, 1))), axis=1),
        'frame_R': None,
        'frame_T': None
    }
    return vo_state

def process_image(file_path):
    sift = cv2.SIFT_create()
    img = cv2.imread(file_path)
    if img is not None:
        _, des = sift.detectAndCompute(img, None)
        return des
    return None

# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()

        # Variables for saving data
        self.count = 0  # Counter for saving images
        self.save_dir = "data/images/"  # Directory to save images to
        self.index = -1

        self.action = []
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        # Load pre-trained codebook for VLAD encoding
        self.codebook = pickle.load(open("codebook.pkl", "rb"))
        # Initialize database for storing VLAD descriptors of FPV
        self.database = []

        # Initialize visual odometry state
        self.vo_state = vo_initialization(320, 240, 92, 92, 160, 120, 0, 0, 0, 0, 0)
        self.img_id = 0
        self.prev_draw_x, self.prev_draw_y = 290, 90
        self.traj_points = []
        self.prev_direction = np.array([20, 0, 20]).T
        self.VPRIndex = 1
        self.target_locations = []
        self.navigate = False
        self.target_traj = []

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    self.last_act ^= self.keymap[event.key]
        if self.last_act is not Action.QUIT:
            self.action.append(self.last_act)

        if self._state and self._state[1] == Phase.NAVIGATION and self.index <= self.goal and self.index < len(self.action) - 1:
            self.index += 1
            return self.action[self.index]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]

        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """
        Display image from database based on its ID using OpenCV
        """
        path = self.save_dir + str(id) + ".jpg"
        img = cv2.imread(path)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    def compute_sift_features(self):
        print('start sift')
        starttime = time.time()
        file_paths = [os.path.join(self.save_dir, filename) for filename in os.listdir(self.save_dir)]
        with Pool() as pool:
            sift_descriptors = pool.map(process_image, file_paths)

        sift_descriptors = [des for des in sift_descriptors if des is not None]
        sift_descriptors = [des for sublist in sift_descriptors for des in sublist]
        endtime = time.time()
        print(f"time used for compute_sift_features: {endtime - starttime}")
        return np.asarray(sift_descriptors)

    def get_VLAD(self, img):
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        _, des = self.sift.detectAndCompute(img, None)
        pred_labels = self.codebook.predict(des)
        centroids = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        for i in range(k):
            if np.sum(pred_labels == i) > 0:
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)

        return VLAD_feature

    def get_neighbor(self, img):
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        _, index = self.tree.query(q_VLAD, 1)
        return index[0][0]

    def pre_nav_compute(self):
        """
        Build BallTree for nearest neighbor search and find the goal ID
        """
        if self.count > 0:
            sift_descriptors = self.compute_sift_features()
            codebook = KMeans(n_clusters=16, init='random', n_init=3, verbose=1).fit(sift_descriptors)
            pickle.dump(codebook, open("codebook.pkl", "wb"))
            tree = BallTree(self.database, leaf_size=100)
            self.tree = tree
            targets = self.get_target_images()
            index = self.get_neighbor(targets[0])
            self.goal = index
            print(f'Goal ID: {self.goal}')

    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()

    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """
        index = self.get_neighbor(self.fpv)
        self.display_img_from_id(index+5, f'Next Best View')
        print(f'Next View ID: {index+5} || Goal ID: {self.goal}')

    def see(self, fpv):
        """
        Set the first-person view input
        """
        if fpv is None:
            return

        if len(fpv.shape) == 3:
            fpv = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)

        if fpv.shape[0] != self.vo_state['cam']['height'] or fpv.shape[1] != self.vo_state['cam']['width']:
            fpv = cv2.resize(fpv, (self.vo_state['cam']['width'], self.vo_state['cam']['height']))

        self.fpv = fpv

        if self.screen is None:
            h, w = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, None].repeat(3, axis=2)  # Convert grayscale to RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        if self._state:
            if self._state[1] == Phase.EXPLORATION:
                save_dir_full = os.path.join(os.getcwd(), self.save_dir)
                save_path = save_dir_full + str(self.count) + ".jpg"
                if not os.path.isdir(save_dir_full):
                    os.mkdir(save_dir_full)
                cv2.imwrite(save_path, fpv)
                VLAD = self.get_VLAD(self.fpv)
                self.database.append(VLAD)
                self.count = self.count + 1
            elif self._state[1] == Phase.NAVIGATION:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]:
                    self.display_next_best_view()

        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()
        
        # Update visual odometry state with new frame only when action is not IDLE
        if self.last_act != Action.IDLE:
            self.vo_state = self.frame_update_step(self.vo_state, fpv, kMinNumFeature, STAGE_DEFAULT_FRAME, STAGE_SECOND_FRAME, STAGE_FIRST_FRAME)
            self.draw_trajectory()
        
    def frame_update_step(self, vo_state, img, kMinNumFeature, STAGE_DEFAULT_FRAME, STAGE_SECOND_FRAME, STAGE_FIRST_FRAME):
        """
        Update the visual odometry system with a new frame.
        Drives the entire visual odometry process by handling each new frame.
        """
        assert (img.ndim == 2 and img.shape[0] == vo_state['cam']['height'] and img.shape[1] == vo_state['cam']['width']), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        vo_state['new_frame'] = img
        if vo_state['frame_stage'] == STAGE_DEFAULT_FRAME:
            vo_state = self.final_PROCESSING(vo_state, kMinNumFeature)
        elif vo_state['frame_stage'] == STAGE_SECOND_FRAME:
            vo_state = self.second_frame_PROCESSING(vo_state, STAGE_DEFAULT_FRAME)
        elif vo_state['frame_stage'] == STAGE_FIRST_FRAME:
            vo_state = self.initial_frame_PROCESSING(vo_state)

        vo_state['last_frame'] = vo_state['new_frame']
        return vo_state

    def initial_frame_PROCESSING(self, vo_state):
        """
        Detect and initialize features in the first frame.
        Establishes the initial set of keypoints for tracking.
        """
        vo_state['px_ref'] = vo_state['detector'].detect(vo_state['new_frame'])
        vo_state['px_ref'] = np.array([x.pt for x in vo_state['px_ref']], dtype=np.float32)
        vo_state['frame_stage'] = STAGE_SECOND_FRAME
        return vo_state

    def second_frame_PROCESSING(self, vo_state, STAGE_DEFAULT_FRAME):
        """
        Process the second frame to establish initial motion estimation.
        Extracts keypoints and matches them with the first frame.
        """
        M, vo_state['px_ref'], vo_state['px_cur'] = self.feature_matching(vo_state['last_frame'], vo_state['new_frame'], vo_state['px_ref'])
        
        if M is None:
            # Homography estimation skipped due to insufficient corresponding points
            if vo_state['px_ref'].shape[0] < kMinNumFeature:
                vo_state['px_cur'] = vo_state['detector'].detect(vo_state['new_frame'])
                vo_state['px_cur'] = np.array([x.pt for x in vo_state['px_cur']], dtype=np.float32)
        else:
            # Homography estimation successful
            if vo_state['px_ref'].shape[0] < kMinNumFeature:
                vo_state['px_cur'] = vo_state['detector'].detect(vo_state['new_frame'])
                vo_state['px_cur'] = np.array([x.pt for x in vo_state['px_cur']], dtype=np.float32)
        
        vo_state['frame_stage'] = STAGE_DEFAULT_FRAME
        vo_state['px_ref'] = vo_state['px_cur']
        return vo_state

    def final_PROCESSING(self, vo_state, kMinNumFeature):
        """
        Core processing for each new frame in the visual odometry sequence.
        Tracks feature points, updates poses, and manages keypoint lifecycle.
        """
        _, vo_state['px_ref'], vo_state['px_cur'] = self.feature_matching(vo_state['last_frame'], vo_state['new_frame'], vo_state['px_ref'])
        E, mask = cv2.findEssentialMat(vo_state['px_cur'], vo_state['px_ref'], focal=vo_state['focal'], pp=vo_state['pp'], method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, vo_state['px_cur'], vo_state['px_ref'], focal=vo_state['focal'], pp=vo_state['pp'])
        vo_state['frame_R'] = R
        vo_state['frame_T'] = t
        if vo_state['px_ref'].shape[0] < kMinNumFeature:
            vo_state['px_cur'] = vo_state['detector'].detect(vo_state['new_frame'])
            vo_state['px_cur'] = np.array([x.pt for x in vo_state['px_cur']], dtype=np.float32)
        vo_state['px_ref'] = vo_state['px_cur']
        return vo_state

    def feature_matching(self, last_frame, new_frame, last_pts):
        """
        Perform feature matching between the last and new frames.
        """
        kp1, des1 = extractor.detectAndCompute(last_frame, None)
        kp2, des2 = extractor.detectAndCompute(new_frame, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # Check if there are enough corresponding points
        if len(pts1) < 4 or len(pts2) < 4:
            return None, last_pts, pts2
        
        M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        return M, pts1, pts2
    
    def draw_trajectory(self):
        """
        Draw the trajectory and display it.
        """
        cur_t = self.vo_state['frame_T']
        cur_R = self.vo_state['frame_R']
        if cur_R is None or cur_t is None:
            return
        self.img_id += 1
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
        draw_x, draw_y = int(x) + 290, int(z) + 290
        traj = np.full((720, 720, 3), [0, 165, 255], dtype=np.uint8)
        cv2.circle(traj, (draw_x, draw_y), 1, (self.img_id * 255 / 4540, 255 - self.img_id * 255 / 4540, 0), 1)
        dir = np.array([20, 0, 20]).T
        if cur_R is not None:
            dir = cur_R @ dir
        self.prev_direction = dir
        end_point_x = draw_x - int(dir[0] * 1)
        end_point_y = draw_y - int(dir[2] * 1)
        cv2.arrowedLine(traj, (draw_x, draw_y), (end_point_x, end_point_y), (0, 0, 255), thickness=2)
        if (draw_x, draw_y) != (self.prev_draw_x, self.prev_draw_y):
            if not self.navigate:
                self.traj_points.append([draw_x, draw_y])
            else:
                self.target_traj.append([draw_x, draw_y])
            self.prev_draw_x, self.prev_draw_y = draw_x, draw_y
        for i in range(1, len(self.traj_points)):
            cv2.line(traj, (self.traj_points[i - 1][0], self.traj_points[i - 1][1]), (self.traj_points[i][0], self.traj_points[i][1]), (255, 125, 0), 2)
        for i in range(1, len(self.target_traj)):
            cv2.line(traj, (self.target_traj[i - 1][0], self.target_traj[i - 1][1]), (self.target_traj[i][0], self.target_traj[i][1]), (204, 255, 255), 2)
        for target in self.target_locations:
            cv2.circle(traj, (int(target[0]), int(target[1])), 1, (0, 0, 255), 5)
        text = "Pose:\nx=%2fm\ny=%2fm\nz=%2fm" % (x, y, z)
        lines = text.split('\n')
        text_x, text_y = 20, 40
        font = cv2.FONT_HERSHEY_COMPLEX
        for line in lines:
            cv2.putText(traj, line, (text_x, text_y), font, 1, (255, 255, 255), 1, 8)
            text_y += 30
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)

    def get_state(self):
        """
        This function is to be invoked by players.
        :return: a tuple of the following items:
            bot_fpv: np.ndarray
            phase: Phase
            step: int
            time: float
            fps: float
            time_left: float
        """
        return self._state

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
