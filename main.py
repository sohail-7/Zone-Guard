from flask import Flask,render_template,request,flash
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from datetime import datetime
import os
import os
import glob
now = datetime.now()

model=YOLO('yolov8s.pt')
app=Flask(__name__,template_folder='templates')
app.config['SECRET_KEY']='Jihihihdsujidsji siisoowow'
app.config['UPLOAD_FOLDER'] = 'static/Image'


@app.route("/",methods=['GET','POST'])
def home():

    if request.method=='POST':
        # text=request.form.get('video_id')
        if 'video' in request.files:
            video = request.files['video']  # Get the file from the POST request

            # Check if a file is selected
            if video.filename != '':
                # video.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{video.filename}'))
                video.save(os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')) 
        


        folder_path = 'static/Image'

        # Check if the folder exists
        if os.path.exists(folder_path):
            # Get a list of PNG files in the folder
            png_files = glob.glob(os.path.join(folder_path, '*.png'))

            # Check if there are any PNG files
            if png_files:
        # Delete each PNG file
                for png_file in png_files:
                    try:
                        os.remove(png_file)
                        print(f"Deleted: {png_file}")
                    except Exception as e:
                        print(f"Error deleting {png_file}: {e}")
                
            else:
                print(f"No PNG files found in '{folder_path}'.")
        else:
            print(f"The folder '{folder_path}' does not exist.")

        # print(text)

        # t = T(text)

        # # Call the run method to execute the code in the T class
        # area = t.run()
        # print(area)
        #---------------------------------


        def print_first_frame_with_points(video_path, width, height):
            #global area
            cap = cv2.VideoCapture(video_path)

            # Check if the video is opened successfully
            if not cap.isOpened():
                print("Error: Could not open the video.")
                return

            ret, frame = cap.read()

            # Resize the frame to the desired width and height
            frame = cv2.resize(frame, (width, height))

            # Define a list to store selected points
            selected_points = []

            # Counter to keep track of selected points
            points_counter = 0
            prompt_text = "Select Four Coordinates"

            # Function to display the prompt text
            def display_prompt():
                cv2.putText(frame, prompt_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('First Frame', frame)

            # Display the initial prompt
            display_prompt()

            # Detect and print four points on the first frame
            def select_points(event, x, y, flags, param):
                nonlocal points_counter

                if event == cv2.EVENT_LBUTTONDOWN and points_counter < 4:
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.imshow('First Frame', frame)
                    selected_points.append((x, y))
                    points_counter += 1

            cv2.imshow('First Frame', frame)
            cv2.setMouseCallback('First Frame', select_points)
            cv2.waitKey(0)  # Wait indefinitely for a key press
            cv2.destroyAllWindows()

            if len(selected_points) == 4:
                print("Selected points:", selected_points)
                return selected_points
            else:
                print("Please select exactly four points.")

            # Release the video capture object
            cap.release()

        # Define the path to your video file
        video_path = 'static/Image/video.mp4'
        # Define the desired width and height for the frame
        frame_width = 1020
        frame_height = 500

        # Call the function to print the first frame with points
        global area
        area=print_first_frame_with_points(video_path, frame_width, frame_height)
        print(area)
        #---------------------------------




        def RGB(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE :  
                colorsBGR = [x, y]
                print(colorsBGR)
        

        cv2.namedWindow('RGB')
        cv2.setMouseCallback('RGB', RGB)

        cap=cv2.VideoCapture(video_path)
        my_file = open("coco1.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")
        #print(class_list)
        count=0
        tracker=Tracker()
        # t=T(video_path)
        # print(t)


        # area=[(54,436),(41,449),(317,494),(317,470)]  
        #import cv2
        
        #area=[(54,436),(41,449),(317,494),(317,470)] 
        
        area_c=set()

        def imgwrite(img):
            import cv2
            import os
            from datetime import datetime

            # Get the current working directory
            current_directory = os.getcwd()

            # Specify the subdirectory where you want to save the image
            image_directory = 'static/Image'

            # Construct the full path to the image directory
            full_image_directory = os.path.join(current_directory, image_directory)

            # Check if the image directory exists, and if not, create it
            if not os.path.exists(full_image_directory):
                os.makedirs(full_image_directory)

                # Get the current date and time
            now = datetime.now()
            current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
            filename = '%s.png' % current_time

            # Your image data (e.g., loaded with cv2.imread)
            # image_data = ...

            # Save the image to the full path
            cv2.imwrite(os.path.join(full_image_directory, filename), img)

            # now = datetime.now()
            # current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
            # filename = '%s.png' % current_time
            
            # cv2.imwrite(os.path.join(r"C:\Users\SOHAIL IBRAHIM\Desktop\in\static\Image",filename), img)
        while True:    
            ret,frame = cap.read()
            if not ret:
                break
            count += 1
            if count % 2 != 0:
                continue


            frame=cv2.resize(frame,(1020,500))

            results=model.predict(frame)
            #   print(results)
            a=results[0].boxes.boxes
            px=pd.DataFrame(a).astype("float")

            list=[]
            for index,row in px.iterrows():

 
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[5])
                c=class_list[d]
                if 'person' in c:
                    list.append([x1,y1,x2,y2])
            
            bbox_idx=tracker.update(list)
            for bbox in bbox_idx:
                x3,y3,x4,y4,id=bbox
                results=cv2.pointPolygonTest(np.array(area,np.int32),((x4,y4)),False)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
                if results>=0:
                    crop=frame[y3:y4,x3:x4]
                    imgwrite(crop)
                    # cv2.imshow(str(id),crop) 
                    area_c.add(id)
            cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
            print(area_c)
            k=len(area_c)
            cv2.putText(frame,str(k),(50,60),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),3)
            cv2.imshow("RGB", frame)
            if cv2.waitKey(1)&0xFF==27:
                break
        cap.release()
        cv2.destroyAllWindows()



    return render_template('base.html')

@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/clr",methods=['GET','POST'])
def clr():
    if request.method=='POST':
        flash('Deleted Successfully',category='success')
        text=request.form.get('text')

        folder_path = 'static/Image'

        # Check if the folder exists
        if os.path.exists(folder_path):
            # Get a list of PNG files in the folder
            png_files = glob.glob(os.path.join(folder_path, '*.png'))

            # Check if there are any PNG files
            if png_files:
        # Delete each PNG file
                for png_file in png_files:
                    try:
                        os.remove(png_file)
                        print(f"Deleted: {png_file}")
                    except Exception as e:
                        print(f"Error deleting {png_file}: {e}")
            else:
                print(f"No PNG files found in '{folder_path}'.")
        else:
            print(f"The folder '{folder_path}' does not exist.")

        def delete_video(video_filename):
            folder_path = 'static/Image'
            video_path = os.path.join(folder_path, video_filename)

            # Check if the file exists before attempting to delete
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"The video '{video_filename}' has been deleted.")
            else:
                print(f"The video '{video_filename}' does not exist.")

            # Usage: Replace 'video_to_delete.mp4' with the name of the video file you want to delete
        delete_video('video.mp4')



    return render_template('clr.html')

@app.route("/ninja/")
def ninja():
    file_list=[]
    basepath="static/Image"
    dir=os.walk(basepath)
    for path,subdirs,files in dir:
        for file in files:
            # temp=path+'/'+  file
            file_list.append(file)
            # print(path,subdirs)
            # print(file)
    return render_template('img.html',hists=file_list)


if __name__=='__main__':
    app.run(debug=True)
