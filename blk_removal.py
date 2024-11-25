from collections import deque
import numpy as np 
from display_image import viewimage
import cv2

#test if the columns are only black pixels

def test_black_column(img,blk_tresh):
    Ly = img.shape[0] #number of lines
    Lx = img.shape[1] #number of columns
    column_index =[]
    i = 0
    m = 1
    while i < Lx and m == 1 :
        test = []
        for j in range(Ly):
            if img[j,i] < blk_tresh:
                test.append(1)
            else:
                test.append(0)
        m = np.mean(test)
        if m == 1:
            column_index.append(i)
        i += 1   
    return column_index

def test_black_column_rverse(img,blk_tresh):
    Ly=img.shape[0] #number of lines
    Lx=img.shape[1] #number of columns
    column_index=[]
    i = Lx - 1
    m = 1 
    while i > 0 and m == 1 : 
        test = []
        for j in range (Ly):
            if img[j,i] < blk_tresh:
                test.append(1)
            else:
                test.append(0)
        m = np.mean(test)
        if m == 1:
            column_index.append(i)
        i -= 1
    return column_index


def blk_column_index(img,blk_tresh):
    return test_black_column(img,blk_tresh) + test_black_column_rverse(img,blk_tresh)

def test_black_line(img,blk_tresh):
    Ly = img.shape[0]
    Lx = img.shape[1]
    line_index = []
    j = 0 
    m = 1
    while j < Ly and m == 1:
        test = []
        for i in range (Lx):
            if img[j,i] < blk_tresh:
                test.append(1)
            else:
                test.append(0)
        m = np.mean(test)
        if m == 1:
            line_index.append(j)
        j += 1
    return line_index

def test_black_line_rverse(img,blk_tresh):
    Ly = img.shape[0]
    Lx = img.shape[1]
    line_index = []
    j = Ly - 1
    m = 1
    while j > 0 and m == 1:
        test = []
        for i in range (Lx):
            if img[j,i] < blk_tresh:
                test.append(1)
            else: 
                test.append(0)
        m = np.mean(test)
        if m == 1:
            line_index.append(j)
        j -= 1
    return line_index

def blk_line_index(img,blk_tresh):
    return test_black_line(img,blk_tresh)+test_black_line_rverse(img, blk_tresh)


#definition of a pixel set.

def create_set(x,y,L,img) :
    s = np.zeros((L,L),dtype=img.dtype)
    coordinates = []
    for i in range (L):
        for j in range (L):
            if 0<=j+y<img.shape[0] and 0<=i+x<img.shape[1]:
                s[j,i]=img[j+y,i+x]
                coordinates.append((y+j, x+i))
    return s, coordinates

def set_black(x,y,L,img,tau):
    s = create_set(x,y,L,img)
    coordinates = s[1]
    s1 = s[0]
    test = []
    for i in range (L):
        for j in range (L):
            if s1[j,i] < tau :
                test.append(1)
            else:
                test.append(0)
    m = np.mean(test)
    if m == 1: 
        return True, coordinates
    else :
        return False, coordinates


def get_coordinates_black_pixel(x,y,img,tau):
    if img[y,x] < tau : 
        return [y,x]
    else :
        return None


def region_growing(x,y, L, img, tau):
    S = set_black(x,y, L, img, tau)
    Ly, Lx = img.shape
    mark = set()  # Utilisation d'un set pour le marquage
    waiting = deque()  # Utilisation de deque pour waiting
    coordinates = []
    #print("size of S", len(S[1]))
    if S[0]:
        waiting.extend(S[1])  # Création d'une copie indépendante de S[1]
        coordinates.extend(S[1])
        #print("size of waiting", len(waiting))
        #print("size of coordinates", len(coordinates))
        for p in S[1]:
            mark.add((p[0], p[1]))
        count = 0
        while waiting and count < 100000:
            tup = waiting.popleft()  # Retire le premier élément efficacement
            #print("LISTE ATTENTE DEBUT BOUCLE", list(waiting))
            j, i = tup  # Déballage pour plus de clarté
            #print("pixel en cours de visite", j, ":", i)

            # Vérification des voisins
            voisins = [(j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1),
                       (j + 1, i + 1), (j + 1, i - 1), (j - 1, i - 1), (j - 1, i + 1)]
            for J, I in voisins:
                if 0 <= I < Lx and 0 <= J < Ly:
                    if img[J, I] < tau and (J, I) not in mark:
                        mark.add((J, I))
                        waiting.append((J, I))
                        coordinates.append((J, I))
                        #print(f"Ajouté à waiting: ({J}, {I})")
                        #print(f"Ajouté à coordinates: ({J}, {I})")

        return coordinates     
    else:
        print("No black frame  in the picture")
        return False

def mask_remove(img,tau,l,x,y):
    Ly = img.shape[0]
    Lx = img.shape[1]
    mask = np.ones_like(img,dtype=np.uint8)
    setting_blk_corner2 = region_growing(x,y,l, img,tau)
    setting_blk_corner1 = region_growing(Lx-x,y,l,img,tau)
    setting_blk_corner3 = region_growing(x,Ly-y,l, img,tau)
    setting_blk_corner4 = region_growing(Lx-x,Ly-y,l, img,tau)
    if not (setting_blk_corner2 and setting_blk_corner1 and setting_blk_corner3 and setting_blk_corner4):
        return mask
    setting_blk = setting_blk_corner2 + setting_blk_corner1 + setting_blk_corner3 + setting_blk_corner4
    for p in setting_blk:
        mask[p[0],p[1]]= 0
    return mask

def Remove(img,tau,l,x,y):
    Ly = img.shape[0]
    Lx = img.shape[1]
    mask = img.copy()
    setting_blk_corner2 = region_growing(x,y,l, img,tau)
    setting_blk_corner1 = region_growing(Lx-x,y,l,img,tau)
    setting_blk_corner3 = region_growing(x,Ly-y,l, img,tau)
    setting_blk_corner4 = region_growing(Lx-x,Ly-y,l, img,tau)
    if not (setting_blk_corner2 and setting_blk_corner1 and setting_blk_corner3 and setting_blk_corner4):
        return mask
    setting_blk = setting_blk_corner2 + setting_blk_corner1 + setting_blk_corner3 + setting_blk_corner4
    for p in setting_blk:
        mask[p[0],p[1]]= 255
    return mask



