from PIL import Image
import cv2
import numpy as np
import pytesseract
import keyboard

vfile = cv2.VideoCapture(1) # 노트북 내장 (기본)웹캠은 0, 외장 웹캠은 1
pytesseract.pytesseract.tersseract_cmd = r'C:\Program Files\Tesseract-OCR'


if vfile.isOpened(): # vfile이 정상적으로 Open 되었나
    while True:
        vret, img = vfile.read() # 제대로 프레임을 읽으면 vret가 True, 실패시 False, img는 읽은 프레임이 나옴

        img2 = img.copy()
        grayimg = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # 영상을 흑백으로 만듦
        blurimg = cv2.GaussianBlur(grayimg, (9, 9), 1) # 가우시안 블러(입력 이미지, 커널사이즈, 표준편차)
        vret1, thr1 = cv2.threshold(blurimg, 140, 255, cv2.THRESH_BINARY)
        # vret: 사용된 임계값, thr1: 출력이미지, cv2.threshold: 이진화(입력 이미지, 문턱값, 최대값, 문턱값 적용 방법)
        # thr2 = Image.fromarray(thr1)
        contours, hierarchy = cv2.findContours(thr1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # dst: 이미지(여기선 생략), contours: 컨투어 정보(x, y좌표 데이터가 한 세트), hierarchy: 컨투어의 상하구조,
        # cv2.findContours: 윤곽선 찾기(contour를 찾을 이미지, contour 추출 모드, contour 근사방법)
        contoursimg = cv2.cvtColor(thr1, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contoursimg, contours, -1, (0, 255, 0), 3)
        # cv2.drawContours: 윤곽선 그림(입력 영상, 컨투어 배열, 인덱스, 색상 값, 선 두께)

        contours_dict = [] # 윤곽선의 x, y, w, h, cx, cy 정보를 담을 빈 리스트 생성

        # 검출된 '모든' 컨투어의 정보를 사전형태로 저장.
        for contour in contours: # 컨투어 왼쪽 위 점, 폭, 높이, 중심점 xy 좌표를 사전 형태로 저장
            x, y, w, h = cv2.boundingRect(contour)
            # v2.boundingRect(): 괄호안의 윤곽선정보를 가지고 직사각형의 정보를 도출
            cv2.rectangle(img2,(x,y), (x+w,y+h), (255,0,180), 3) # 감지한 모든 사각형 그리기

            contours_dict.append({
            # 'contour':contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx':x+(w/2),
            'cy':y+(h/2)
            }) # contours_dict라는 리스트의 각 원소들은 사전형태!


        MIN_AREA, MAX_AREA = 500, 10000 # 최소 넓이 (픽셀 기준)
        MIN_WIDTH, MIN_HEIGHT = 3, 5 # 최소 폭, 높이
        MIN_RATIO, MAX_RATIO = 0.25, 10 # 최소,최대 비율

        possible_contours = []
        possible = []

        count = 0


        # ----------------------------------------------------------기준에 따라 컨투어를 필터링---------------------------------------------------------------------
        for d in contours_dict:
            # 넓이, 종횡비 기준에 따라서 딕셔너리 contours_dict를 필터링후,
            area = d['w']*d['h']
            ratio = d['w']/d['h']

            if MAX_AREA > area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                if 0.7 < ratio < 1.3 and 30 < area < 60:
                    # 각 숫자 안의 사각형이 검출되지 않도록 특정 종횡비 및 넓이의 사각형 제거, ex) 8안에 rect 2개
                    continue
                else:
                    # 아닐경우, [x, y, w, h, cx, cy]의 리스트로 만들어서 possible_contours 리스트에 추가
                    # d['idx'] = count
                    # count += 1
                    x1 = d['x']
                    y1 = d['y']
                    w1 = d['w']
                    h1 = d['h']
                    cx1 = d['cx']
                    cy1 = d['cy']
                    possible_contours.append([x1, y1, w1, h1, cx1, cy1])

        #--------------------------------------------------------번호판 영역인지 판별하는 부분------------------------------------------------------------
        MAX_DIAG_MULTIPLYER = 1.5 # 상자 사이의 거리
        MAX_ANGLE_DIFF = 70
        MAX_AREA_DIFF = 0.5
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.6
        MIN_N_MATCHED = 3

        result = []
        
        
        # 걸러낸 사각형들중 번호판 문자 부분만 걸러냄
        def find_chars(contour_list): # 번호판 알고리즘을 통하여 번호판 부위만 걸러내기
            matched_result = [] # 함수에서 최종적으로 걸러진 사각형들을 저장할 리스트 생성

            for d1 in contour_list: # 윤곽선 좌표1
                matched_contours = []
                for d2 in contour_list: # 윤곽선 좌표2
                    if d1 == d2: # 좌표1과 좌표2가 같을 경우(같은 점일경우), 제외
                        continue

                    dx = abs(d1[4] - d2[4])
                    dy = abs(d1[5] - d2[5])

                    distance = np.sqrt(dx**2+dy**2)
                    digonal_length = np.sqrt(d1[2]**2+d1[3]**2)

                    if dx == 0:
                        angle_diff = 90
                    else:
                        angle_diff = np.degrees(np.arctan(dy/dx))

                    area_diff = abs(d1[2]*d1[3] - d2[2]*d2[3]) / (d1[2]*d1[3])
                    width_diff = abs(d1[2] - d2[2]) / d1[2]
                    height_diff = abs(d1[3] - d2[3]) / d1[3]

                    if distance < digonal_length*MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                        matched_contours.append(d2)

                matched_contours.append(d1)

                if len(matched_contours) < MIN_N_MATCHED:
                    continue

                for n in range(0, len(matched_contours)):
                    matched_result.append(matched_contours[n])

            return matched_result

        result = sorted(find_chars(possible_contours)) # x좌표가 작은수부터 큰수로 가도록 리스트요소를 정렬
        result2 = []

        for n in range(0,len(result)): # result에서 중복되는 것을 걸러냄
            if result[n] in result2:
                continue
            result2.append(result[n])

        for n in range(0,len(result2)):
            cv2.rectangle(contoursimg, (result2[n][0], result2[n][1]), \
                          (result2[n][0] + result2[n][2], result2[n][1] + result2[n][3]), \
                          (0,0,255), 3) # 필터링된 사각형 보는 코드(번호판 영역)


        if vret:
            cv2.imshow('webcam', img2)
            # cv2.imshow('graycam', grayimg)
            cv2.imshow('blurcam', blurimg)
            # cv2.imshow('bincam', thr1)
            cv2.imshow('contourscam', contoursimg)
            if not result2 == []:
                width = result2[-1][0] + result[-1][2] - result2[0][0]
                height = result2[0][3]
                pts1 = np.float32([(result2[0][0]-3, result2[0][1]-3),
                                   (result2[-1][0] + result2[-1][2]+6, result2[-1][1]-3),
                                   (result2[0][0]-3, result2[0][1] + result2[0][3]+3),
                                   (result2[-1][0] + result2[-1][2]+6, result2[-1][1] + result2[-1][3]+3)])
                pts2 = np.float32([[3, 3], [width - 3, 3], [3, height - 3], [width - 3, height - 3]])
                warp = cv2.getPerspectiveTransform(pts1, pts2)
                ROI = cv2.warpPerspective(thr1, warp, (width,height))
                cv2.imshow('ROI', ROI)
                str = pytesseract.image_to_string(ROI, lang='eng', config=('-l eng --oem 3 --psm 4'))
                print(str)

            if cv2.waitKey(1) == ord('q'): # 아무키나 누르면 파일 닫음
                break
        else:
            print('Frame is abnormal.')
            break
    else:
        print('Unable to open file.')


    # print(possible_contours)
    # print("\n\n")
    # print(result)
    # print("\n\n")
    # print(result2)
    # print("\n\n")
    # print(str)
    # print("\n\n")

    vfile.release() # 오픈한 vfile객체를 해제
    cv2.destroyAllWindows() # 화면에 나타난 윈도우를 종료함.
