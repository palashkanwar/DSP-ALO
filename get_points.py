# Import the required modules
import cv2

def run(im, mode, for_pedestrian = False,):
    im_disp = im.copy()
    im_draw = im.copy()
    window_name = "Select objects to be tracked here."
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name,455,0)
    cv2.imshow(window_name, im_draw)

    if mode and for_pedestrian:
        ped_sex = []

    # List containing top-left and bottom-right to crop the image.
    pts_1 = []
    pts_2 = []

    rects = []
    run.mouse_down = False

    #For removing the bug that kept on adding pedestrains when
    #Pedestrian sex was not supplied
    run.adding_sex = True

    def callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN and run.adding_sex == True:
            pts_1.append((x, y))
            run.mouse_down = True

        elif event == cv2.EVENT_LBUTTONUP and run.mouse_down == True:
            run.mouse_down = False
            pts_2.append((x, y))

            if mode and for_pedestrian:
                print ("object selected\n")
                print ("press 'm' or 'f' on the window to assign sex: ")
                run.adding_sex = False
                if cv2.waitKey(-1) & 0xFF == ord('m'):
                    print ("Added sex 'm' ")
                    ped_sex.append('m')
                    run.adding_sex = True
                elif cv2.waitKey(-1) & 0xFF == ord('f'):
                    print ("Added sex 'f' ")
                    ped_sex.append('f')
                    run.adding_sex = True
                #print ped_sex
            print ("Object added succesfully\n")

            print ("\nYou can add more objects. \nPress 'd' to delete the last selected objects.") \
                  ("\nPress 's' to save and continue with the present selection \n")

        elif event == cv2.EVENT_MOUSEMOVE and run.mouse_down == True:
            im_draw = im.copy()
            cv2.rectangle(im_draw, pts_1[-1], (x, y), (255,255,255), 3)
            cv2.imshow(window_name, im_draw)
            cv2.moveWindow(window_name, 455, 0)

    #print "Press and release mouse around the object to be tracked. \n You can also select multiple objects."
    cv2.setMouseCallback(window_name, callback)

    print ("Press key `s` any time to save and \n           continue with the selected points.")
    print ("Press key `d` to discard the last object selected.")
    print ("Press key `r` to resume the tracking.\n")

    while True:
        # Draw the rectangular boxes on the image
        window_name_2 = "Objects to be tracked."
        for pt1, pt2 in zip(pts_1, pts_2):
            rects.append([pt1[0],pt2[0], pt1[1], pt2[1]])
            cv2.rectangle(im_disp, pt1, pt2, (255, 255, 255), 3)
        # Display the cropped images
        cv2.namedWindow(window_name_2, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name_2, 910, 0)
        cv2.imshow(window_name_2, im_disp)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            print ("Saved.")
            point= [(tl + br) for tl, br in zip(pts_1, pts_2)]
            corrected_point=check_point(point)

            ##Return tuple for pedestrain
            if for_pedestrian and mode:
                return corrected_point, ped_sex
            else:
                return corrected_point

        elif key== ord('r'):
            if for_pedestrian and mode:
                return "QUIT", "QUIT"
            else:
                return "QUIT"


        elif key == ord('d'):
            # Press ket `d` to delete the last rectangular region
            if run.mouse_down == False and pts_1:
                pts_1.pop()
                pts_2.pop()
                if for_pedestrian and mode:
                    try:
                        ped_sex.pop()
                    except IndexError:
                        pass
                im_disp = im.copy()
                print ("last object deleted")
            else:
                print ("No object to delete.")

def check_point(points):
    out=[]
    for point in points:
        #to find min and max x coordinates
        if point[0]<point[2]:
            minx=point[0]
            maxx=point[2]
        else:
            minx=point[2]
            maxx=point[0]
        #to find min and max y coordinates
        if point[1]<point[3]:
            miny=point[1]
            maxy=point[3]
        else:
            miny=point[3]
            maxy=point[1]
        out.append((minx,miny,maxx,maxy))

    return out


