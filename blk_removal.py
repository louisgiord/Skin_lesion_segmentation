from collections import deque
import numpy as np 
import  cv2
import matplotlib
from otsu_seg import viewimage, view2images

# Load the image
img=cv2.imread ("images_test/img1.jpg")

#Check is the image is correctly charged
if img is None:
    raise FileNotFoundError("image not found")

#Transform the image to gray scale
img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

#parameters

T = 30

#viewimage(img_gray)
#viewimage(img)
#view2images(img, img_gray) pb of size between the two images

# generate an image with a black column in the middle and elsewhere white pixels
im_black = np.zeros((200,200))
def region_growing(l, L, img, tau):
    S = ALLsets_black_test(l, L, img, tau)
    Ly, Lx = img.shape
    mark = set()  # Utilisation d'un set pour le marquage
    waiting = deque()  # Utilisation de deque pour waiting
    coordinates = []

    if S[0]:
        waiting.extend(S[1])  # Création d'une copie indépendante de S[1]
        coordinates.extend(S[1])
        
        for p in S[1]:
            mark.add((p[0], p[1]))

        count = 0
        while waiting and count < 100000:
            tup = waiting.popleft()  # Retire le premier élément efficacement
            print("LISTE ATTENTE DEBUT BOUCLE", list(waiting))
            i, j = tup  # Déballage pour plus de clarté
            print("pixel en cours de visite", j, ":", i)

            # Vérification des voisins
            voisins = [(j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1),
                       (j + 1, i + 1), (j + 1, i - 1), (j - 1, i - 1), (j - 1, i + 1)]
            for J, I in voisins:
                if 0 <= I < Lx and 0 <= J < Ly:
                    if img[J, I] < tau and (J, I) not in mark:
                        mark.add((J, I))
                        waiting.append((J, I))
                        coordinates.append((J, I))
                        print(f"Ajouté à waiting: ({J}, {I})")
                        print(f"Ajouté à coordinates: ({J}, {I})")

        return coordinates     
    else:
        print("No black frame in the picture")


# Création d'une image noire de 200x200 pixels
im_black = np.zeros((200, 200))

# Définition des dimensions de la bordure noire
border_size = 70

# Définition des indices pour le carré blanc au milieu
start_index = border_size
end_index = 200 - border_size

# Remplissage du carré blanc au milieu
im_black[start_index:end_index, start_index:end_index] = 255

viewimage(im_black)

#definition of a pixel set.

def create_set(x,y,L,img) :
    s = np.zeros((L,L),dtype=img.dtype)
    coordinates = []
    for i in range (L):
        for j in range (L):
            s[j,i]=img[y+j,x+i]
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


#function verifying if the corners of the image are black

def ALLsets_black_test(l,L,img,tau):
    Ly = img.shape[0]
    Lx = img.shape[1]
    set1_test = set_black(l,l,L,img,tau)
    #set4_test = set_black(Lx-1-l,Ly-1-l,L,img,tau)
    test = set1_test[0] #and set3_test[0] and set2_test[0] and set4_test[0]
    coordinates = set1_test[1] #+ set3_test[1] + set2_test[1] + set4_test[1]
    return test, coordinates

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

    if S[0]:
        waiting.extend(S[1])  # Création d'une copie indépendante de S[1]
        coordinates.extend(S[1])
        
        for p in S[1]:
            mark.add((p[0], p[1]))

        count = 0
        while waiting and count < 100000:
            tup = waiting.popleft()  # Retire le premier élément efficacement
            print("LISTE ATTENTE DEBUT BOUCLE", list(waiting))
            i, j = tup  # Déballage pour plus de clarté
            print("pixel en cours de visite", j, ":", i)

            # Vérification des voisins
            voisins = [(j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1),
                       (j + 1, i + 1), (j + 1, i - 1), (j - 1, i - 1), (j - 1, i + 1)]
            for J, I in voisins:
                if 0 <= I < Lx and 0 <= J < Ly:
                    if img[J, I] < tau and (J, I) not in mark:
                        mark.add((J, I))
                        waiting.append((J, I))
                        coordinates.append((J, I))
                        print(f"Ajouté à waiting: ({J}, {I})")
                        print(f"Ajouté à coordinates: ({J}, {I})")

        return coordinates     
    else:
        print("No black frame in the picture")

setting = region_growing(10,10,5,im_black,20)

for p in setting:
        im_black[p[0],p[1]]= 255

viewimage(im_black)


viewimage(img_gray)

Ly = img_gray.shape[0]
Lx = img_gray.shape[1]

setting_blk_corner1 = region_growing(10,10,5, img_gray,120)
setting_blk_corner2 = region_growing(Lx-10,10,5, img_gray,60)
setting_blk_corner3 = region_growing(10,Ly-10,5, img_gray,60)
setting_blk_corner4 = region_growing(Lx-10,Ly-10,5, img_gray,60)
setting_blk = setting_blk_corner1 + setting_blk_corner2 + setting_blk_corner3 + setting_blk_corner4


for p in setting_blk:
        img_gray[p[0],p[1]]= 255



viewimage(img_gray)





