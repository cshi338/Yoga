import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 


def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
		return True 
	return False

model  = load_model("model.h5")
label = np.load("labels.npy")



holistic = mp.solutions.pose
holis = holistic.Pose(static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5)
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


while True:
	lst = []
	lst1 = []

	_, frm = cap.read()

	window = np.zeros((940,940,3), dtype="uint8")

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
	frm = cv2.blur(frm, (4,4))
	if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
		for i in res.pose_landmarks.landmark:
			lst.append(i.x - res.pose_landmarks.landmark[0].x)
			lst.append(i.y - res.pose_landmarks.landmark[0].y)
		lst = np.array(lst).reshape(1,-1)
		
		
		# Yoga Pose Prediction using Model
		p = model.predict(lst)
		pred = label[np.argmax(p)]
		#print(pred) 
		#print(p[0][np.argmax(p)]) 

		# Extract Features from Pose
		nose = [res.pose_landmarks.landmark[holistic.PoseLandmark.NOSE].x, 
				res.pose_landmarks.landmark[holistic.PoseLandmark.NOSE].y, 
				res.pose_landmarks.landmark[holistic.PoseLandmark.NOSE].z]
		left_shoulder = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_SHOULDER.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_SHOULDER.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_SHOULDER.value].z])
		right_shoulder = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_SHOULDER.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_SHOULDER.value].z])
		left_elbow = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_ELBOW.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_ELBOW.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_ELBOW.value].z])
		right_elbow = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ELBOW.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ELBOW.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ELBOW.value].z])
		left_wrist = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_WRIST.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_WRIST.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_WRIST.value].z])
		right_wrist= np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_WRIST.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_WRIST.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_WRIST.value].z])
		left_hip = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_HIP.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_HIP.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_HIP.value].z])
		right_hip = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_HIP.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_HIP.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_HIP.value].z])
		left_knee = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_KNEE.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_KNEE.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_KNEE.value].z])
		right_knee = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_KNEE.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_KNEE.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_KNEE.value].z])
		left_ankle = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_ANKLE.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_ANKLE.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_ANKLE.value].z])
		right_ankle = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ANKLE.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ANKLE.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_ANKLE.value].z])
		left_index = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_INDEX.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_INDEX.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_INDEX.value].z])
		right_index = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_INDEX.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_INDEX.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_INDEX.value].z])
		left_heel = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_HEEL.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_HEEL.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_HEEL.value].z])
		right_heel = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_HEEL.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_HEEL.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_HEEL.value].z])
		left_foot_index = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_FOOT_INDEX.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_FOOT_INDEX.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.LEFT_FOOT_INDEX.value].z])
		right_foot_index = np.array([res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
				res.pose_landmarks.landmark[holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].z])
	
		
		# Calculate Mid Points of Shoulder, hip and knee

		shldr_mid = (left_shoulder + right_shoulder)/2
		hip_mid = (left_hip + right_hip)/2
		knee_mid = (left_knee + right_knee)/2
		wrst_mid = (left_wrist + right_wrist)/2
		ankl_mid = (left_ankle + right_ankle)/2
		
		feedback = ["Perfect Pose"]
		
		
		if (p[0][np.argmax(p)]) > 0.70:
			cv2.putText(window, pred[:-2], (200,50),cv2.FONT_ITALIC, 1.6, (0,255,0),2)
		
			# Yoga Pose Correction for Different Poses:
			if pred == 'Tadasana':

				knee_dist = abs(left_knee[0]-right_knee[0])
				hip_dist = abs(left_hip[0]-right_hip[0])
				feedback.append(" Put Your Feet Closer") if knee_dist > hip_dist else feedback.append("")
				#hand_depth = abs(left_ankle[2]-right_ankle[2])
				
				left_hip_shldr = left_hip - left_shoulder
				left_wrist_shldr = left_wrist - left_shoulder
				
				cosine_angle_left = np.dot(left_hip_shldr, left_wrist_shldr) / (np.linalg.norm(left_hip_shldr) * np.linalg.norm(left_wrist_shldr))
				angle_left_hand = np.degrees(np.arccos(cosine_angle_left))
				print(angle_left_hand)
				feedback.append(" Bring Your Right Hand Closer") if angle_left_hand > 60 else feedback.append("")
				

				right_hip_shldr = right_hip - right_shoulder
				right_wrist_shldr = right_wrist - right_shoulder

				
				cosine_angle_right = np.dot(right_hip_shldr, right_wrist_shldr) / (np.linalg.norm(right_hip_shldr) * np.linalg.norm(right_wrist_shldr))
				angle_right_hand = np.degrees(np.arccos(cosine_angle_right))
				print(angle_right_hand)
				feedback.append(" Bring Your Left Hand Closer") if angle_right_hand > 60 else feedback.append("")
				pred_out= 'Tadasana'
				
				if (feedback[1] != '') or (feedback[2] != '') or (feedback[3] != ''):
					cv2.putText(window, "1.  "+feedback[1]+"  "+"2.  "+feedback[2]+"  "+"3.  "+feedback[3],(100,150), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
				else:
					cv2.putText(window, " Perfect Pose", (200,200),cv2.FONT_ITALIC, 1.6, (0,255,0),2)
								
				
				
				
				#print(knee_dist , hip_dist)	
			
			if (pred == 'Adho_Mukha_Svanasana_L') or (pred =='Adho_Mukha_Svanasana_R'):
				mid_wrst_shldr = wrst_mid - shldr_mid
				mid_hip_shldr = wrst_mid - shldr_mid
				
				cosine_angle_wr_sdr_hip = np.dot(mid_wrst_shldr , mid_hip_shldr) / (np.linalg.norm(mid_wrst_shldr) * np.linalg.norm(mid_hip_shldr))
				angle_wr_sdr_hip = np.degrees(np.arccos(cosine_angle_wr_sdr_hip))
				feedback.append(" Spread Your Hands Further") if angle_wr_sdr_hip < 150 else feedback.append("")
				mid_kni_hip = knee_mid - hip_mid
				mid_shldr_hip = shldr_mid - hip_mid
				
				cosine_angle_sdr_hip_kni = np.dot(mid_kni_hip, mid_shldr_hip) / (np.linalg.norm(mid_kni_hip) * np.linalg.norm(mid_shldr_hip))
				angle_sdr_hip_kni = np.degrees(np.arccos(cosine_angle_sdr_hip_kni))
				feedback.append(" Spread More from Hips") if angle_sdr_hip_kni < 80 else feedback.append("")
				mid_hip_kni = hip_mid - knee_mid
				mid_ankl_kni = ankl_mid - knee_mid
				
				cosine_angle_hip_kni_ankl = np.dot(mid_hip_kni, mid_ankl_kni) / (np.linalg.norm(mid_hip_kni) * np.linalg.norm(mid_ankl_kni))
				angle_hip_kni_ankl = np.degrees(np.arccos(cosine_angle_hip_kni_ankl))
				feedback.append(" Straighten your legs from knee") if angle_hip_kni_ankl < 160 else feedback.append("")			

				if (feedback[1] != '') or (feedback[2] != '') or (feedback[3] != ''):
					cv2.putText(window, "1.  "+feedback[1]+"  "+"2.  "+feedback[2]+"  "+"3.  "+feedback[3],(100,150), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
				else:
					cv2.putText(window, " Perfect Pose", (200,200),cv2.FONT_ITALIC, 1.6, (0,255,0),2)


			if (pred == 'Phalakasana-R') or (pred =='Phalakasana-L'):
				
				mid_wrst_shldr = wrst_mid - shldr_mid
				mid_hip_shldr = wrst_mid - shldr_mid
				
				cosine_angle_wr_sdr_hip = np.dot(mid_wrst_shldr , mid_hip_shldr) / (np.linalg.norm(mid_wrst_shldr) * np.linalg.norm(mid_hip_shldr))
				angle_wr_sdr_hip = np.degrees(np.arccos(cosine_angle_wr_sdr_hip))
				feedback.append("Pull Upper Body Forward") if angle_wr_sdr_hip > 80 else feedback.append("")
				
			
				mid_kni_hip = knee_mid - hip_mid
				mid_shldr_hip = shldr_mid - hip_mid
				
				cosine_angle_sdr_hip_kni = np.dot(mid_kni_hip, mid_shldr_hip) / (np.linalg.norm(mid_kni_hip) * np.linalg.norm(mid_shldr_hip))
				angle_sdr_hip_kni = np.degrees(np.arccos(cosine_angle_sdr_hip_kni))
				feedback.append(" Straighten Body from Hips") if angle_sdr_hip_kni < 160 else feedback.append("")
				print('ok2')
				
				mid_hip_kni = hip_mid - knee_mid
				mid_ankl_kni = ankl_mid - knee_mid
				
				cosine_angle_hip_kni_ankl = np.dot(mid_hip_kni, mid_ankl_kni) / (np.linalg.norm(mid_hip_kni) * np.linalg.norm(mid_ankl_kni))
				angle_hip_kni_ankl = np.degrees(np.arccos(cosine_angle_hip_kni_ankl))
				feedback.append(" Straighten your legs from knee") if angle_hip_kni_ankl < 160 else feedback.append("")			
				
				if (feedback[1] != '') or (feedback[2] != '') or (feedback[3] != ''):
					cv2.putText(window, "1.  "+feedback[1]+"  "+"2.  "+feedback[2]+"  "+"3.  "+feedback[3],(100,150), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
				else:
					cv2.putText(window, " Perfect Pose", (200,200),cv2.FONT_ITALIC, 1.6, (0,255,0),2)
			
			if (pred == 'Ardha_Chandrasana_L') or (pred =='Ardha_Chandrasana_R'):
				mid_rankl_hip = right_ankle - hip_mid
				mid_lankl_hip = left_ankle - hip_mid
				cosine_angle_rankl_hip_lankl = np.dot(mid_rankl_hip , mid_lankl_hip) / (np.linalg.norm(mid_rankl_hip) * np.linalg.norm(mid_lankl_hip))
				angle_rankl_hip_lankl = np.degrees(np.arccos(cosine_angle_rankl_hip_lankl))

				feedback.append("Further Raise Your Leg Up") if angle_rankl_hip_lankl < 70 else feedback.append("")
				
				mid_lwrst_shldr = right_wrist- shldr_mid
				mid_rwrst_shldr = left_wrist- shldr_mid
				
				cosine_angle_lwrst_shldr_rwrst = np.dot(mid_lwrst_shldr , mid_rwrst_shldr) / (np.linalg.norm(mid_lwrst_shldr) * np.linalg.norm(mid_rwrst_shldr))
				angle_lwrst_shldr_rwrst = np.degrees(np.arccos(cosine_angle_lwrst_shldr_rwrst))
				feedback.append("Align Both Hands") if angle_lwrst_shldr_rwrst <150 else feedback.append("")
				
				if pred == 'Ardha_Chandrasana_L':
					y_ardh_ch_l = abs(right_index[1]-right_foot_index[1])/right_foot_index[1]	
					feedback.append("Align Your Upper Body ") if y_ardh_ch_l < 0.9 else feedback.append("")
				if pred == 'Ardha_Chandrasana_R':
					y_ardh_ch_r = abs(left_index[1]-left_foot_index[1])/left_foot_index[1]
					print(y_ardh_ch_r)
					feedback.append("Align Your Upper Body ") if y_ardh_ch_r < 0.9 else feedback.append("")
				
				if (feedback[1] != '') or (feedback[2] != '') or (feedback[3] != ''):
					
					cv2.putText(window, "1.  "+feedback[1]+"  "+"2.  "+feedback[2]+"  "+"3.  "+feedback[3],(100,150), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
				else:
					cv2.putText(window, " Perfect Pose", (200,200),cv2.FONT_ITALIC, 1.6, (0,255,0),2)

			if (pred == 'Vrksasana-L') or (pred =='Vrksasana-R'):
		
				feedback.append("Join Your Hands Over Head") if abs(left_index[1]-right_index[1]) > 0.05 else feedback.append("")
				feedback.append("Join Your Hands") if (left_wrist[1]>left_shoulder[1]) or (right_wrist[1]>right_shoulder[1]) else feedback.append("")
				if (pred == 'Vrksasana-L'):
					#if (left_ankle[1]<right_knee[1]) or (right_ankle[1]< left_knee[1]:
					feedback.append("Raise Your Foot close to Knee") if (left_ankle[1] > right_knee[1]) else feedback.append("")
				else:
					feedback.append("Raise Your Foot close to Knee") if (right_ankle[1] > left_knee[1]) else feedback.append("")
				feedback.append("Align Your Knee with Body") if (abs(right_knee[2]-left_knee[2]) > 0.2) else feedback.append("")

				if (feedback[1] != '') or (feedback[2] != '') or (feedback[3] != ''):
					
					cv2.putText(window, "1.  "+feedback[1]+"  "+"2.  "+feedback[2]+"  "+"3.  "+feedback[3] +"  "+"4.  "+feedback[4],(100,150), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
				else:
					cv2.putText(window, " Perfect Pose", (200,200),cv2.FONT_ITALIC, 1.2, (0,255,0),2)
			
			if (pred == 'Vasisthasana_R') or (pred=='Vasisthasana_L'):

				vasi_mid_lwrst_shldr = right_wrist- shldr_mid
				vasi_mid_rwrst_shldr = left_wrist- shldr_mid
				
				cosine_angle_vasi_lwrst_shldr_rwrst = np.dot(vasi_mid_lwrst_shldr , vasi_mid_rwrst_shldr) / (np.linalg.norm(vasi_mid_lwrst_shldr) * np.linalg.norm(vasi_mid_rwrst_shldr))
				angle_vasi_lwrst_shldr_rwrst = np.degrees(np.arccos(cosine_angle_vasi_lwrst_shldr_rwrst))
				feedback.append("Align Your Hands ") if angle_vasi_lwrst_shldr_rwrst < 120 else feedback.append("")
				
				
				vasi_mid_knee_hip = knee_mid - hip_mid 
				vasi_mid_shldr_hip = shldr_mid - hip_mid
				shldr_mid = (left_shoulder + right_shoulder)/2
				cosine_angle_vasi_kni_hip_shldr = np.dot(vasi_mid_knee_hip  , vasi_mid_shldr_hip) / (np.linalg.norm(vasi_mid_knee_hip ) * np.linalg.norm(vasi_mid_shldr_hip))
				angle_vasi_lwrst_kni_hip_shldr = np.degrees(np.arccos(cosine_angle_vasi_kni_hip_shldr))
		
				feedback.append("Align Your Hips") if angle_vasi_lwrst_kni_hip_shldr < 150 else feedback.append("")
				if (feedback[1] != '') or (feedback[2] != ''):
					
					cv2.putText(window, "1.  "+feedback[1]+"  "+"2.  "+feedback[2],(100,150), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
				else:
					cv2.putText(window, " Perfect Pose", (200,200),cv2.FONT_ITALIC, 1.2, (0,255,0),2)
			if (pred == 'Virabhadrasana_II_L') or (pred=='Virabhadrasana_II_R'):
				virab_mid_lwrst_shldr = right_wrist- shldr_mid
				virab_mid_rwrst_shldr = left_wrist- shldr_mid
				
				cosine_angle_virab_lwrst_shldr_rwrst = np.dot(virab_mid_lwrst_shldr , virab_mid_rwrst_shldr) / (np.linalg.norm(virab_mid_lwrst_shldr) * np.linalg.norm(virab_mid_rwrst_shldr))
				angle_virab_lwrst_shldr_rwrst = np.degrees(np.arccos(cosine_angle_virab_lwrst_shldr_rwrst))
				feedback.append("Align Your Hands ") if angle_virab_lwrst_shldr_rwrst <120 else feedback.append("")
				print(angle_virab_lwrst_shldr_rwrst)
				
				virab_lni_midhip = left_knee-hip_mid
				virab_rni_midhip = right_knee-hip_mid

				cosine_angle_virab_lkni_midhip_rni = np.dot(virab_lni_midhip, virab_rni_midhip) / (np.linalg.norm(virab_lni_midhip) * np.linalg.norm(virab_rni_midhip))
				angle_virab_lkni_midhip_rni = np.degrees(np.arccos(cosine_angle_virab_lkni_midhip_rni))
				feedback.append("Stretch Your Legs ") if angle_virab_lkni_midhip_rni< 70 else feedback.append("")
				feedback.append("Align Your Hands ") if (abs(left_wrist[2]-left_knee[2])> 0.01) or (abs(left_wrist[2]-left_knee[2])>0.01) else feedback.append("")
				print(abs(left_wrist[2]-left_knee[2]),  abs(right_wrist[2]-right_knee[2]))

				if (feedback[1] != '') or (feedback[2] != '') or (feedback[2] != ''):
					
					cv2.putText(window, "1.  "+feedback[1]+"  "+"2.  "+feedback[2]+"  "+"3.  "+feedback[3],(100,150), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
				else:
					cv2.putText(window, " Perfect Pose", (200,200),cv2.FONT_ITALIC, 1.2, (0,255,0),2)
		else:
			cv2.putText(window, "Asana is either wrong not trained" , (200,110),cv2.FONT_ITALIC, 1, (0,0,255),3)
				
				
				#if (p[0][np.argmax(p)] > 0.70) and (len(feedback)>1):
					#cv2.putText(window, pred_out, (200,50),cv2.FONT_ITALIC, 1.6, (0,255,0),2)
					#cv2.putText(window, "1.  "+feedback[1]+"  "+"2.  "+feedback[2]+"  "+"3.  "+feedback[3],(100,150), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
					#cv2.putText(window, feedback[2],(100,180), cv2.FONT_ITALIC, 0.6, (0,255,0),2)
					#cv2.putText(window, feedback[3],(100,210), cv2.FONT_ITALIC, 0.6, (0,255,0),2)

				

		

		#feedback_out =""
		#for i in range(len(feedback)):
			#feedback_out += feedback[i]

		
	#else: 
		#cv2.putText(frm, "Make Sure Full body visible", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),3)

		
	
	drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
							connection_drawing_spec=drawing.DrawingSpec(color=(255,255,255), thickness=6 ),
							 landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), circle_radius=3, thickness=3))


	window[320:800, 270:910, :] = cv2.resize(frm, (640, 480))

	cv2.imshow("window", window)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break

