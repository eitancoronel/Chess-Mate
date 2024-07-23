"""
Chess Robot Automation Script from the project Chess Mate
Youtube explanation video: https://www.youtube.com/watch?v=HJsMiX6PEWk&t=0s&ab_channel=EitanCoronel
Author: Eitan Coronel
Description: This script automates the movements of a chess-playing robot using RoboDK, Stockfish, and computer vision
techniques. The robot can capture and move chess pieces on a physical board, detect changes in board state via a camera,
and play against the Stockfish chess engine.
"""

from robodk.robolink import Robolink, ITEM_TYPE_ROBOT
from robodk import *  # robodk robotics toolbox
import time
from pythonosc import udp_client
import chess
import chess.engine
import cv2
import numpy as np

# Error message for invalid moves
error_message = "Enter your move (e.g., 'p e2 e4')"

# Define the size of each square in mm
SQUARE_SIZE = 41


class ChessRobot:
    def __init__(self, robot, client):
        """
        Initialize the ChessRobot with the given robot and UDP client.
        """
        self.robot = robot
        self.client = client
        self.BOARD_SQUARES = 8
        self.val = {
            'P': 18,
            'Q': 28,
            'K': 31,
            'N': 22,
            'R': 21,
            'B': 25
        }
        self.board = chess.Board()
        self.stockfish_path = "C:\\Users\\eitancoronel\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam.")

    def square_to_position(self, letter, number):
        """
        Convert a square coordinate to robot position in mm.
        Args:
            letter (str): File letter of the chessboard square.
            number (str): Rank number of the chessboard square.
        Returns:
            tuple: x and y positions in mm.
        """
        y = ord(letter) - ord('A')
        x = int(number) - 1
        x_pos = 225 + x * SQUARE_SIZE
        y_pos = 170 - y * SQUARE_SIZE
        return x_pos, y_pos

    def move_piece(self, src_letter, src_number, dest_letter, dest_number, val):
        """
        Move a piece on the chessboard using the robot arm.
        Args:
            src_letter (str): Source file letter.
            src_number (str): Source rank number.
            dest_letter (str): Destination file letter.
            dest_number (str): Destination rank number.
            val (int): Lift height value for the piece.
        """
        src_x, src_y = self.square_to_position(src_letter, src_number)
        dest_x, dest_y = self.square_to_position(dest_letter, dest_number)

        self.move_robot_to_position(src_x, src_y, 80)
        self.move_robot_to_position(src_x, src_y, val)
        self.robot.setDO(11, 1)  # Pick up the piece
        time.sleep(1)
        self.move_robot_to_position(src_x, src_y, 80)
        self.move_robot_to_position(dest_x, dest_y, 80)
        self.move_robot_to_position(dest_x, dest_y, val + 1)
        self.robot.setDO(11, -1)  # Release the piece
        time.sleep(2)
        self.move_robot_to_position(dest_x, dest_y, 80)
        self.move_robot_to_home()

    def move_robot_to_position(self, x, y, z):
        """
        Move the robot to a specific position.
        Args:
            x (int): X-coordinate.
            y (int): Y-coordinate.
            z (int): Z-coordinate.
        """
        pose = TxyzRxyz_2_Pose([x, y, z, pi, 0, -pi])
        self.robot.MoveL(pose)

    def move_robot_to_home(self):
        """
        Move the robot to the home position.
        """
        self.move_robot_to_position(360, -250, 90)

    @staticmethod
    def split_square(square_name):
        """
        Split a square name (e.g., 'e4') into file and rank.
        Args:
            square_name (str): The square name.
        Returns:
            tuple: File and rank of the square.
        """
        return square_name[0], square_name[1]

    @staticmethod
    def print_board_with_coordinates(board):
        """
        Print the chessboard with coordinates.
        Args:
            board (chess.Board): The chess board object.
        """
        print("  a b c d e f g h")
        print(" +----------------")
        for rank in range(7, -1, -1):
            row = f"{rank + 1}|"
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece:
                    row += piece.symbol() + " "
                else:
                    row += ". "
            print(row + f"|{rank + 1}")
        print(" +----------------")
        print("  a b c d e f g h")

    @staticmethod
    def capture_image(cap, image_name):
        """
        Capture an image from the camera and save it.
        Args:
            cap (cv2.VideoCapture): The video capture object.
            image_name (str): The name to save the image as.
        Returns:
            str: Path to the saved image.
        """
        ret, frame = cap.read()
        if ret:
            image_path = f'{image_name}.jpg'
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved as '{image_path}'")
            return image_path
        else:
            print("Error: Failed to capture image.")
            return None

    @staticmethod
    def detect_and_highlight_objects(image, rows, cols):
        """
        Detect blue and red objects in the image, highlight them, and return the board state.
        Args:
            image (numpy.ndarray): Input image.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
        Returns:
            list: Annotated board state as a 2D list.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.add(red_mask1, red_mask2)

        num_labels_blue, labels_blue, stats_blue, centroids_blue = cv2.connectedComponentsWithStats(blue_mask)
        num_labels_red, labels_red, stats_red, centroids_red = cv2.connectedComponentsWithStats(red_mask)

        height, width = image.shape[:2]
        dy, dx = height // rows, width // cols

        detected_boxes = set()
        blue_boxes = set()
        red_boxes = set()

        board = [["." for _ in range(cols)] for _ in range(rows)]

        def highlight_and_print_objects(num_labels, centroids, stats, color_bgr, color_boxes, symbol):
            """
            Highlight and print the detected objects on the image and board.
            Args:
                num_labels (int): Number of detected objects.
                centroids (numpy.ndarray): Centroids of the detected objects.
                stats (numpy.ndarray): Statistics of the detected objects.
                color_bgr (tuple): BGR color value for drawing.
                color_boxes (set): Set to store detected boxes.
                symbol (str): Symbol to represent the detected objects on the board.
            """
            for i in range(1, num_labels):
                x = int(centroids[i][0])
                y = int(centroids[i][1])
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                center = (x, y)
                radius = int(max(width, height) / 2)
                cv2.circle(image, center, radius, color_bgr, 2)

                row = y // dy
                col = x // dx

                chess_row = chr(ord('H') - row)
                chess_col = 8 - col
                box = f'{chess_col}{chess_row}'

                if box not in detected_boxes:
                    detected_boxes.add(box)
                    color_boxes.add(box)
                    board[row][col] = symbol

        highlight_and_print_objects(num_labels_blue, centroids_blue, stats_blue, (255, 255, 0), blue_boxes, 'B')
        highlight_and_print_objects(num_labels_red, centroids_red, stats_red, (0, 255, 255), red_boxes, 'R')

        print("  8 7 6 5 4 3 2 1")
        print(" +----------------")
        for rank in range(8):
            row = f"{chr(ord('H') - rank)}|"
            for file in range(8):
                row += board[rank][file] + " "
            print(row + f"|{chr(ord('H') - rank)}")
        print(" +----------------")
        print("  8 7 6 5 4 3 2 1")
        return board

    @staticmethod
    def draw_grid(image, rows, cols):
        """
        Draw a grid on the image.
        Args:
            image (numpy.ndarray): Input image.
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
        Returns:
            numpy.ndarray: Annotated image with a grid.
        """
        height, width = image.shape[:2]
        dy, dx = height // rows, width // cols

        smaller_dx = int(dx * 0.9)

        for i in range(0, height, dy):
            cv2.line(image, (0, i), (width, i), (0, 255, 255), 2)

        for col in range(cols):
            x = col * dx
            if 4 <= col <= 7:
                smaller_x = x + smaller_dx
                if smaller_x < width:
                    cv2.line(image, (smaller_x, 0), (smaller_x, height), (0, 255, 255), 2)
            if col == 4:
                cv2.line(image, (x, 0), (x, height), (0, 255, 255), 2)
            elif col < 4:
                cv2.line(image, (x, 0), (x, height), (0, 255, 255), 2)

        return image

    @staticmethod
    def crop_and_warp(image_path, coords):
        """
        Crop and warp the image based on the given coordinates.
        Args:
            image_path (str): Path to the input image.
            coords (list): Coordinates for cropping and warping.
        Returns:
            numpy.ndarray: Cropped and warped image.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image from {image_path}")
            return None
        pts = np.array(coords, dtype=np.float32)
        width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
        height = max(np.linalg.norm(pts[0] - pts[2]), np.linalg.norm(pts[1] - pts[3]))
        dst_pts = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))

        return warped

    @staticmethod
    def save_image(image, output_path):
        """
        Save the image to the specified output path.
        Args:
            image (numpy.ndarray): Image to be saved.
            output_path (str): Path to save the image.
        """
        if cv2.imwrite(output_path, image):
            print(f"Image saved to {output_path}")
        else:
            print(f"Error: Unable to save image to {output_path}")

    @staticmethod
    def show_image(image, scale_factor=1.5):
        """
        Display the image in a window until any key is pressed.
        Args:
            image (numpy.ndarray): The image to display.
            scale_factor (float): Factor by which to scale the image.
        """
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Cropped Image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def image_cropper(self, image_path, board_status):
        """
        Crop the image based on predefined coordinates and return the cropped image.
        Args:
            image_path (str): Path to the input image.
            board_status (str): Status label for the board (e.g., 'prev' or 'curr').
        Returns:
            numpy.ndarray: Cropped image.
        """
        output_path = f"cropped_{board_status}_image.jpg"
        coords = [(75, 25), (470, 27), (56, 432), (489, 429)]
        warped_image = self.crop_and_warp(image_path, coords)
        if warped_image is not None:
            self.save_image(warped_image, output_path)
            return warped_image
        return None

    def compare_boards(self, prev_board, curr_board, stockfish_board, detection_color):
        """
        Compare the previous and current boards, and print the move made by the player.
        Args:
            prev_board (list): The board state from the previous image.
            curr_board (list): The board state from the current image.
            stockfish_board (chess.Board): The Stockfish board to keep track of the game.
            detection_color (str): The color of the player making the move ('R' or 'B').
        Returns:
            str: The move made in standard algebraic notation.
        """
        rows, cols = len(prev_board), len(prev_board[0])
        prev_diff = [["." for _ in range(cols)] for _ in range(rows)]
        curr_diff = [["." for _ in range(cols)] for _ in range(rows)]

        for row in range(rows):
            for col in range(cols):
                if prev_board[row][col] != curr_board[row][col]:
                    if prev_board[row][col] != ".":
                        prev_diff[row][col] = prev_board[row][col]
                    if curr_board[row][col] != ".":
                        curr_diff[row][col] = curr_board[row][col]

        B_prev_diffs = [(chr(ord('H') - row), 8 - col, prev_diff[row][col])
                        for row in range(rows) for col in range(cols) if prev_diff[row][col] == 'B']
        B_curr_diffs = [(chr(ord('H') - row), 8 - col, curr_diff[row][col])
                        for row in range(rows) for col in range(cols) if curr_diff[row][col] == 'B']
        R_prev_diffs = [(chr(ord('H') - row), 8 - col, prev_diff[row][col])
                        for row in range(rows) for col in range(cols) if prev_diff[row][col] == 'R']
        R_curr_diffs = [(chr(ord('H') - row), 8 - col, curr_diff[row][col])
                        for row in range(rows) for col in range(cols) if curr_diff[row][col] == 'R']

        final_move = None
        capture_box = None
        for r_prev in R_prev_diffs:
            for b_curr in B_curr_diffs:
                if r_prev[:2] == b_curr[:2]:
                    capture_box = r_prev[:2]
                    from_square = f"{B_prev_diffs[0][0].lower()}{B_prev_diffs[0][1]}"
                    to_square = f"{b_curr[0].lower()}{b_curr[1]}"
                    break
            if capture_box:
                break

        if capture_box:
            print(f"Capture move at {capture_box[0]}{capture_box[1]}")
            piece = stockfish_board.piece_at(chess.parse_square(from_square))
            if piece:
                piece_notation = piece.symbol().upper() if piece.symbol().upper() != 'P' else ''
                capture_move = f"{piece_notation}{from_square}x{to_square}"
                print(f"Capture move at {capture_box[0]}{capture_box[1]}")
                print(f"Stockfish capture notation: {capture_move}")
                final_move = capture_move
        else:
            for b_prev in B_prev_diffs:
                from_square = f"{b_prev[0].lower()}{b_prev[1]}"
                for b_curr in B_curr_diffs:
                    to_square = f"{b_curr[0].lower()}{b_curr[1]}"
                    move = f"{from_square}{to_square}"
                    try:
                        move_obj = stockfish_board.parse_san(move)
                        stockfish_board.push(move_obj)
                        print(f":simple move:{move}")
                        final_move = move
                    except ValueError:
                        continue

        print("B_prev:", B_prev_diffs)
        print("B_curr:", B_curr_diffs)
        print("R_prev:", R_prev_diffs)
        print("R_curr:", R_curr_diffs)
        print("Differences in previous board:")
        for diff in B_prev_diffs + R_prev_diffs:
            print(f"{diff[2]} piece at {diff[0]}{diff[1]}")
        print("Differences in current board:")
        for diff in B_curr_diffs + R_curr_diffs:
            print(f"{diff[2]} piece at {diff[0]}{diff[1]}")

        return final_move

    def run(self):
        """
        Main function to initialize the board, Stockfish engine, and process images.
        """
        self.move_robot_to_home()
        while not self.board.is_game_over():
            self.print_board_with_coordinates(self.board)
            if self.board.turn == chess.BLACK:
                self.process_player_move()
            else:
                self.process_stockfish_move()
        self.end_game()

    def process_player_move(self):
        """
        Process the player's move by capturing images, detecting changes, and updating the board.
        """
        prev_image_path = self.capture_image(self.cap, 'prev')
        if prev_image_path is None:
            exit()

        prev_image = self.image_cropper(prev_image_path, 'prev')
        if prev_image is not None:
            prev_image_with_grid = self.draw_grid(prev_image, self.BOARD_SQUARES, self.BOARD_SQUARES)
            prev_board = self.detect_and_highlight_objects(prev_image_with_grid, self.BOARD_SQUARES, self.BOARD_SQUARES)
            self.show_image(prev_image_with_grid)

            input("Press Enter to capture the second image (curr)...")
            curr_image_path = self.capture_image(self.cap, 'curr')
            if curr_image_path is None:
                exit()

            curr_image = self.image_cropper(curr_image_path, 'curr')
            if curr_image is not None:
                curr_image_with_grid = self.draw_grid(curr_image, self.BOARD_SQUARES, self.BOARD_SQUARES)
                curr_board = self.detect_and_highlight_objects(curr_image_with_grid, self.BOARD_SQUARES,
                                                               self.BOARD_SQUARES)
                self.show_image(curr_image_with_grid)

            move = self.compare_boards(prev_board, curr_board, self.board, 'R')

            try:
                move_obj = self.board.parse_san(move)
                self.board.push(move_obj)
            except ValueError:
                print("################################### GOOD #################################################")
                pass

    def process_stockfish_move(self):
        """
        Process Stockfish's move and update the board accordingly.
        """
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        chosen_move = result.move
        from_square = chess.square_name(chosen_move.from_square)
        to_square = chess.square_name(chosen_move.to_square)
        src_letter, src_number = self.split_square(from_square)
        dest_letter, dest_number = self.split_square(to_square)

        if self.board.is_capture(chosen_move):
            print(f"Captured piece at {to_square}, need to move to garbage!!!!!!")
            dest_x, dest_y = self.square_to_position(dest_letter, dest_number)
            garbage = TxyzRxyz_2_Pose([360, -250, 90, pi, 0, -pi])
            self.move_robot_to_position(dest_x, dest_y, 80)
            piece_symbol = self.board.piece_at(chosen_move.to_square).symbol().upper()
            self.move_robot_to_position(dest_x, dest_y, self.val[piece_symbol])
            self.robot.setDO(11, 1)
            time.sleep(1)
            self.move_robot_to_position(dest_x, dest_y, 80)
            self.robot.MoveL(garbage)
            self.move_robot_to_position(360, -250, self.val[piece_symbol] + 5)
            self.robot.setDO(11, 0)
            self.move_robot_to_position(380, -250, self.val[piece_symbol] + 5)
            self.move_robot_to_position(360, -250, 80)
            print("Captured piece moved to garbage.")

        piece = self.board.piece_at(chosen_move.from_square)
        move_str = f"{piece.symbol()} {from_square} to {to_square}"
        print("Stockfish move:", move_str)
        self.board.push(chosen_move)
        self.move_piece(src_letter, src_number, dest_letter, dest_number, self.val[piece.symbol()])

    def end_game(self):
        """
        End the game by displaying the result and cleaning up resources.
        """
        print("Game Over")
        print("Result: ", self.board.result())
        self.engine.quit()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RDK = Robolink()
    robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise Exception('No robot selected or available')

    ip = "127.0.0.1"
    port = 5005
    client = udp_client.SimpleUDPClient(ip, port)

    chess_robot = ChessRobot(robot, client)
    chess_robot.run()
