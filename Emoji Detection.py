import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceReaction():

    def __init__(self):
        self.right_eye_close = False
        self.left_eye_close = False
        self.landmark = None
        self.image = None
        self.desired_size_x = 64
        self.desired_size_y = 64
        self.reaction_image_pos_x = 10
        self.reaction_image_pos_y = 550
        self.face_info = dict()

    def closeness(self):
        mark1,mark2 = 234, 323
        landmark = np.array(self.landmark).reshape((-1, 1, 2))
        markUpper = landmark[mark1][0]
        markLower = landmark[mark2][0]
        return math.hypot(markUpper[0]-markLower[0],markUpper[1]-markLower[1])

    def drawCircle(self,mark):
        try:
            landmark = np.array(self.landmark).reshape((-1, 1, 2))
            markPoint = landmark[mark][0]
            cv2.circle(self.image, (markPoint[0], markPoint[1]), 15, (255, 0, 0), cv2.FILLED)
        except:
            pass

    def drawLine(self,mark1,mark2):
        landmark = np.array(self.landmark).reshape((-1, 1, 2))
        try:
            markUpper = landmark[mark1][0]
            markLower = landmark[mark2][0]
            cv2.line(self.image, tuple(markUpper), tuple(markLower), (255, 0, 0), 2)
            cv2.circle(self.image, (markLower[0], markLower[1]), 1, (255, 0, 255), cv2.FILLED)
            cv2.circle(self.image, (markUpper[0], markUpper[1]), 1, (0, 0, 255), cv2.FILLED)
        except:
            pass

    def distance(self,mark1,mark2):
        landmark = np.array(self.landmark).reshape((-1, 1, 2))
        try:
            markUpper = landmark[mark1][0]
            markLower = landmark[mark2][0]
            dist = math.hypot(markUpper[0] - markLower[0], markUpper[1] - markLower[1]) / self.closeness()
            return dist,True
        except:
            return -1,False

    def checkTeath(self):
        threshold_distr = 0.13
        threshold_distl = 0.09
        mark1,mark2 = 13 , 14
        distInfo = self.distance(mark1,mark2)
        # self.drawLine(mark1,mark2)
        if distInfo[1]:
            return True if threshold_distl< self.distance(mark1,mark2)[0] < threshold_distr else False
        else:
            return False

    def checkSad(self):
        pass


    def checkRightEyeClose(self):
        threshold_dist = 0.026
        mark1,mark2 = 159 , 145
        distInfo = self.distance(mark1,mark2)
        # self.drawLine(mark1,mark2)
        if distInfo[1]:
            return True if self.distance(mark1,mark2)[0] < threshold_dist else False
        else:
            return False

    def checkLeftEyeClose(self):
        threshold_dist = 0.026
        mark1,mark2 = 386 , 374
        distInfo = self.distance(mark1,mark2)
        # self.drawLine(mark1,mark2)
        if distInfo[1]:
            return True if self.distance(mark1,mark2)[0] < threshold_dist else False
        else:
            return False

    def checkKiss(self):
        threshold_dist = 0.026
        mark1,mark2 = 78 , 13
        distInfo = self.distance(mark1,mark2)
        # self.drawLine(mark1,mark2)
        if distInfo[1]:
            return True if self.distance(mark1,mark2)[0] < threshold_dist else False
        else:
            return False

    def checkWow(self):
        threshold_dist = 0.19
        mark1,mark2 = 13 , 14
        mark3,mark4,mark5,mark6 = 61,24,254,291
        threshold_dist1 = 0.49
        distInfo3 = self.distance(mark3,mark4)
        distInfo4 = self.distance(mark5,mark6)
        distInfo = self.distance(mark1,mark2)
        # self.drawLine(mark1,mark2)
        if distInfo[1] and distInfo3[1] and distInfo4[1]:
            case1 = True if self.distance(mark1,mark2)[0] > threshold_dist else False
            case2 = True if self.distance(mark3,mark4)[0] > threshold_dist1 else False
            case3 = True if self.distance(mark5,mark6)[0] > threshold_dist1 else False
            return True if case1+case2+case3 else False
        else:
            return False


    def checkSmile(self):
        threshold_dist = 0.42
        mark1,mark2,mark3,mark4 = 57,24,254,287
        distInfo1 = self.distance(mark1,mark2)
        distInfo2 = self.distance(mark3,mark4)
        # self.drawLine(mark1,mark2)
        # self.drawLine(mark3,mark4)
        if distInfo1[1] and distInfo2[1] and self.checkLeftEyeClose()+self.checkRightEyeClose() == False:
            case1 =  True if self.distance(mark1,mark2)[0] < threshold_dist else False
            case2 =  True if self.distance(mark3,mark4)[0] < threshold_dist else False
            return True if case1 and case2 else False
        else:
            return False

    def checkSmirk(self):
        threshold_dist = 0.42
        mark1,mark2,mark3,mark4 = 61,24,254,291
        distInfo1 = self.distance(mark1,mark2)
        distInfo2 = self.distance(mark3,mark4)
        if distInfo1[1] or distInfo2[1] and self.checkTeath() == False\
                and (self.checkLeftEyeClose()+self.checkRightEyeClose()) == False:
            # conditions for the only one side smirk
            case1 =  True if self.distance(mark1,mark2)[0] < threshold_dist else False
            case2 =  True if self.distance(mark3,mark4)[0] < threshold_dist else False
            case = case1*case2
            if case1+case2 == False:
                return False
            return False if case else True
        else:
            return False

    def detector(self):
        self.face_info['right_eye_closed'] = self.checkRightEyeClose()
        self.face_info['left_eye_closed'] = self.checkLeftEyeClose()
        self.face_info['smirk'] = self.checkSmirk()
        self.face_info['smile'] = self.checkSmile()
        self.face_info['teeth'] = self.checkTeath()
        self.face_info['wow'] = self.checkWow()
        self.face_info['kiss'] = self.checkKiss()
        if self.checkWow() :
            if self.checkLeftEyeClose() or self.checkRightEyeClose():
                self.face_info['jibe'] = True
        else:
            self.face_info['jibe'] = False
        return self.face_info


    def setReaction(self,img,reactionImage):
        self.toolImage = cv2.resize(reactionImage, (self.desired_size_x, self.desired_size_y))
        img[self.reaction_image_pos_x:self.reaction_image_pos_x + self.desired_size_x, self.reaction_image_pos_y:self.reaction_image_pos_y + self.desired_size_y, :] = reactionImage[
                                                                                           :self.desired_size_x,
                                                                                           :self.desired_size_y, :]




normal = cv2.imread("emojis/normal.png")
leftEyeClosed = cv2.imread("emojis/wink.png")
rightEyeClosed = cv2.flip(leftEyeClosed,1)
smirk = cv2.imread("emojis/smirk.png")
smile = cv2.imread("emojis/happy.png")
teethsmile = cv2.imread("emojis/teethsmile.png")
wow = cv2.imread("emojis/wow.png")
jibe = cv2.imread("emojis/jibe.png")


faceReaction = FaceReaction()

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = []
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                for landmark in face.landmark:
                    x = landmark.x
                    y = landmark.y

                    shape = image.shape
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    landmarks.append([relative_x, relative_y])

        faceReaction.landmark = landmarks
        faceReaction.image = image
        faceInfo = faceReaction.detector()


        if faceInfo['jibe']:
            faceReaction.setReaction(image, jibe)
        elif faceInfo["right_eye_closed"]:
            faceReaction.setReaction(image, rightEyeClosed)
        elif faceInfo["left_eye_closed"]:
            faceReaction.setReaction(image, leftEyeClosed)
        elif faceInfo['teeth']:
            faceReaction.setReaction(image,teethsmile)
        elif faceInfo['smile']:
            faceReaction.setReaction(image, smile)
        elif faceInfo['wow']:
            faceReaction.setReaction(image,wow)
        elif faceInfo['smirk']:
            faceReaction.setReaction(image,smirk)
        else:
            faceReaction.setReaction(image, normal)


        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
