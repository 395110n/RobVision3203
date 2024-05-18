# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2
from multiprocessing import Pool
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from multiprocessing import Pool
import time
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
        self.auto_nav = False
        self.action = []
        # Initialize SIFT detector
        # SIFT stands for Scale-Invariant Feature Transform
        self.sift = cv2.SIFT_create()
        # Load pre-trained codebook for VLAD encoding
        # If you do not have this codebook comment the following line
        # You can explore the maze once and generate the codebook (refer line 181 onwards for more)
        self.codebook = pickle.load(open("codebook.pkl", "rb"))
        # Initialize database for storing VLAD descriptors of FPV
        self.database = []
        self.last_pos = None
        
        self.position_history = []
        self.pos_dict = {}
        self.goal = None
        self.constant = 90/37
        self.position = np.zeros(2)
        self.direction = np.array([0,1])
        self.position_history = [np.zeros(2)]
        self.walking = 0
        self.eventType = None
        self.pos_dict_nav = {}
        self.position_history_nav = []
        
    def rotate_vector(self, vector, angle):
        angle_rad = np.deg2rad(angle)

        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector
    
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
                    self.eventType = event.key
                    self.walking = self.count
                    self.prevFPV = self.fpv
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                self.walking = self.count - self.walking
                print("self.walking: ", self.walking)
                displacement = self.walking*self.direction 
                match self.eventType:
                    case pygame.K_UP:
                        self.position += displacement
                    case pygame.K_DOWN:
                        self.position -= displacement
                    case pygame.K_LEFT:
                        angle = -((self.walking * self.constant)%360)
                        print("left turn angle: ", angle)
                        self.direction = self.rotate_vector(self.direction, angle)
                        print(self.direction)
                    case pygame.K_RIGHT:
                        angle = (self.walking * self.constant)%360
                        print("right turn angle: ", angle)
                        self.direction = self.rotate_vector(self.direction, angle)
                        print(self.direction)
                
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
                    
        # print(self.position)
        if self._state:
            self.draw_positions()
        if self.last_act is not Action.QUIT:
            self.action.append(self.last_act)
        
        if self._state and self.auto_nav and self._state[1] == Phase.NAVIGATION and self.index<=self.goal and self.index < len(self.action) - 1:
            self.index+=1
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
        # """
        # Compute SIFT features for images in the data directory
        # """
        print('start sift')
        starttime = time.time()
        file_paths = [os.path.join(self.save_dir, filename) for filename in os.listdir(self.save_dir)] 
        # 使用进程池并行处理图像特征提取
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
        # We use a SIFT in combination with VLAD as a feature extractor as it offers several benefits
        # 1. SIFT features are invariant to scale and rotation changes in the image
        # 2. SIFT features are designed to capture local patterns which makes them more robust against noise
        # 3. VLAD aggregates local SIFT descriptors into a single compact representation for each image
        # 4. VLAD descriptors typically require less memory storage compared to storing the original set of SIFT
        # descriptors for each image. It is more practical for storing and retrieving large image databases efficicently.

        # Pass the image to sift detector and get keypoints + descriptions
        # Again we only need the descriptors
        _, des = self.sift.detectAndCompute(img, None)
        # We then predict the cluster labels using the pre-trained codebook
        # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_
        # Get the number of clusters from the codebook
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        # Loop over the clusters
        for i in range(k):
            # If the current cluster label matches the predicted one
            if np.sum(pred_labels == i) > 0:
                # Then, sum the residual vectors (difference between descriptors and cluster centroids)
                # for all the descriptors assigned to that clusters
                # axis=0 indicates summing along the rows (each row represents a descriptor)
                # This way we compute the VLAD vector for the current cluster i
                # This operation captures not only the presence of features but also their spatial distribution within the image
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        # Apply power normalization to the VLAD feature vector
        # It takes the element-wise square root of the absolute values of the VLAD feature vector and then multiplies 
        # it by the element-wise sign of the VLAD feature vector
        # This makes the resulting descriptor robust to noice and variations in illumination which helps improve the 
        # robustness of VPR systems
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))
        # Finally, the VLAD feature vector is normalized by dividing it by its L2 norm, ensuring that it has unit length
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)

        return VLAD_feature

    def get_neighbor(self, img):
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        # Get the VLAD feature of the image
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        # This function returns the index of the closest match of the provided VLAD feature from the database the tree was created
        distances, indices = self.tree.query(q_VLAD, 10)
        res = [(dis, ind) for dis, ind in zip(distances[0], indices[0])]
        return res

    def pre_nav_compute(self):
        """
        Build BallTree for nearest neighbor search and find the goal ID
        """
        # If this function is called after the game has started
        if self.count > 0:
            # below 3 code lines to be run only once to generate the codebook
            # Compute sift features for images in the database
            print("in prev_nav_compute")
            sift_descriptors = self.compute_sift_features()

            # KMeans clustering algorithm is used to create a visual vocabulary, also known as a codebook,
            # from the computed SIFT descriptors.
            # n_clusters = 64: Specifies the number of clusters (visual words) to be created in the codebook. In this case, 64 clusters are being used.
            # init='k-means++': This specifies the method for initializing centroids. 'k-means++' is a smart initialization technique that selects initial 
            # cluster centers in a way that speeds up convergence.
            # n_init=10: Specifies the number of times the KMeans algorithm will be run with different initial centroid seeds. The final result will be 
            # the best output of n_init consecutive runs in terms of inertia (sum of squared distances).
            # The fit() method of KMeans is then called with sift_descriptors as input data. 
            # This fits the KMeans model to the SIFT descriptors, clustering them into n_clusters clusters based on their feature vectors

            # TODO: try tuning the function parameters for better performance
            codebook = KMeans(n_clusters = 64, init='random', n_init=3, verbose=1).fit(sift_descriptors)
            pickle.dump(codebook, open("codebook.pkl", "wb"))


            # Build a BallTree for fast nearest neighbor search
            # We create this tree to efficiently perform nearest neighbor searches later on which will help us navigate and reach the target location
            
            # TODO: try tuning the leaf size for better performance
            tree = BallTree(self.database, leaf_size=60)
            with open('ball_tree_model.pkl', 'wb') as f:
                pickle.dump(tree, f)
            self.tree = tree

            # Get the neighbor nearest to the front view of the target image and set it as goal
            targets = self.get_target_images()
            index = self.get_neighbor(targets[0])[0][1]
            self.goal = index
            print(f'Goal ID: {self.goal}')

    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        self.position = np.zeros(2)
        self.direction = np.array([0,1])
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """
        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        matches = self.get_neighbor(self.fpv)
        
        if self.last_pos is None:
            next_best_view_index = matches[0][1]
        else:
            # get index that is closest to the goal, the distance is calculated as the difference between the current index and the goal index
            # the index with the smallest difference is the next best view
            # however the difference of index between the next best view and self.last_pos should be smaller than 100
            # which means the next best view should be close to the last view
            next_best_view_index = min(matches, key=lambda x: abs(x[1] - self.goal) if abs(x[1] - self.last_pos) < 100 else 1000)[1]
        self.last_pos = next_best_view_index
        
        if next_best_view_index > self.goal:
            next_best_view_index -= 4
        else:
            next_best_view_index += 5
            
        # Display the next best view
        self.display_img_from_id(next_best_view_index, 'Next Best View')
        
        # Display the next best view id along with the goal id to understand how close/far we are from the goal
        print(f'Next View ID: {next_best_view_index} || Goal ID: {self.goal}')

    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv
        
        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
            
        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?

                # Get full absolute save path
                save_dir_full = os.path.join(os.getcwd(),self.save_dir)
                save_path = save_dir_full + str(self.count) + ".jpg"
                # Create path if it does not exist
                if not os.path.isdir(save_dir_full):
                    os.mkdir(save_dir_full)
                # Save current FPV
                cv2.imwrite(save_path, fpv)

                # Get VLAD embedding for current FPV and add it to the database
                VLAD = self.get_VLAD(self.fpv)
                self.database.append(VLAD)
                self.count = self.count + 1
                self.position_history.append(self.position.copy())
                self.pos_dict[self.count] = self.position.copy()
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()
                self.count = self.count + 1
                self.position_history_nav.append(self.position.copy())
                self.pos_dict_nav[self.count] = self.position.copy()

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()
        
        

    def draw_positions(self):
        # Create a blank image for the trajectory
        traj = np.full((720, 720, 3), [0, 165, 255], dtype=np.uint8)
        
        # Calculate the center of the image for drawing the arrow
        center_x = traj.shape[1] // 2
        center_y = traj.shape[0] // 2
        curr_pos = self.position
        arrow_end = curr_pos + self.direction * 20
        cv2.arrowedLine(traj, (int(center_x + curr_pos[0]), int(center_y + curr_pos[1])),
                        (int(center_x + arrow_end[0]), int(center_y + arrow_end[1])),
                        (255, 0, 0), thickness=2, tipLength=0.5)

        # Draw the arrow representing the current direction
        # Draw the trajectory path
        if self._state[1] == Phase.EXPLORATION:
            length = len(self.position_history)
        else:
            length = len(self.position_history) - 1
        for i in range(1, length):
            # Calculate start and end points of the line segment
            start_point = (int(center_x + self.position_history[i - 1][0]), int(center_y + self.position_history[i - 1][1]))
            end_point = (int(center_x + self.position_history[i][0]), int(center_y + self.position_history[i][1]))
            # Draw the line segment
            cv2.line(traj, start_point, end_point, (255, 128, 0), 2)
        
        for i in range(1, len(self.position_history_nav)):
            # Calculate start and end points of the line segment
            start_point = (int(center_x + self.position_history_nav[i - 1][0]), int(center_y + self.position_history_nav[i - 1][1]))
            end_point = (int(center_x + self.position_history_nav[i][0]), int(center_y + self.position_history_nav[i][1]))
            # Draw the line segment
            cv2.line(traj, start_point, end_point, (0, 255, 0), 2)

        # Draw the first point in red
        first_point = (int(center_x + self.position_history[0][0]), int(center_y + self.position_history[0][1]))
        cv2.circle(traj, first_point, 2, (0, 0, 255), -1)  # -1 indicates filled circle
        
        if self._state[1] == Phase.EXPLORATION:
            curr_point = (int(center_x + self.position_history[-1][0]), int(center_y + self.position_history[-1][1]))
            cv2.circle(traj, curr_point, 5, (0,255,0), -1)  # -1 indicates filled circle
        else :
            curr_point = (int(center_x + self.position_history_nav[-1][0]), int(center_y + self.position_history_nav[-1][1]))
            cv2.circle(traj, curr_point, 5, (0,255,0), -1)
        if self.goal:
            # Draw the last point in green
            last_point = (int(center_x + self.pos_dict[self.goal][0]), int(center_y + self.pos_dict[self.goal][1]))
            cv2.circle(traj, last_point, 5, (255, 0, 0), -1)

        

        # Display the trajectory image
        cv2.imshow('Trajectory', traj)
        # Wait for 1 ms to allow for the image to be displayed properly
        cv2.waitKey(1)

if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())