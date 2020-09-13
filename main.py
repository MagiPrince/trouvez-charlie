#Alan Turing 16 yo is cache.png

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc


def wm(N):
    w = np.exp(-2*np.pi*1j/N)

    nx, ny = (N, N)
    x = np.linspace(0, N-1, nx)
    y = np.linspace(0, N-1, ny)

    xm, ym = np.meshgrid(x, y)

    matrice_puissance = np.multiply(xm, ym)

    return np.power(w, matrice_puissance)

def tfd(f):
    N = len(f)

    matrice = wm(N)

    f_chapeau = np.dot(matrice, f)

    return f_chapeau

def tdf2(matrice):
    horizontale = len(matrice[0])
    verticale = len(matrice)

    matricex = wm(horizontale)
    matricey = wm(verticale)

    f_chapeau = np.dot(matricey, matrice)

    f_chapeau_chapeau = np.transpose(np.dot(matricex, np.transpose(f_chapeau)))

    return f_chapeau_chapeau
    


def itfd(f_chapeau):
    #transformée inversee
    N = len(f_chapeau)

    itfdm = np.linalg.inv(wm(N))

    f = np.dot(itfdm,f_chapeau)
    return f


def itfd2(matrice_chapeau_chapeau):
    #transformée inverese 2 dim
    horizontale = len(matrice_chapeau_chapeau[0])
    verticale = len(matrice_chapeau_chapeau)

    matricex = np.linalg.inv(wm(horizontale))
    matricey = np.linalg.inv(wm(verticale))

    f_chapeau = np.dot(matricey, matrice_chapeau_chapeau)

    f = np.transpose(np.dot(matricex, np.transpose(f_chapeau)))

    return f

def compression(image, taux):
    img, tf = noise_filter(image,taux,taux) 

    horizontale = len(tf[0])
    verticale = len(tf)
    fichier = open("image.david", "w")
    fichier.write(str(horizontale))
    fichier.write(" ")
    fichier.write(str(verticale))
    fichier.write(" ")

    counter = 0

    for i in range(horizontale):
        for j in range(verticale):
            if(tf[j][i] == 0):
                counter += 1
            elif((counter > 0) and (tf[j][i] != 0)):
                fichier.write("0 ")
                fichier.write(str(counter))
                fichier.write(" ")
                fichier.write(str(tf[j][i]))
                fichier.write(" ")
                counter = 0
            else:
                fichier.write(str(tf[j][i]))
                fichier.write(" ")

    fichier.close()

def decompression(image):

    array = []
    val = ""

    with open(image) as f:
        while True:
            c = f.read(1)
            if not c:
                break
            if(c == " "):
                array.append(complex(val))
                val = ""
            else:
                val += c

    horizontal = int(array[0].real)
    vertical = int(array[1].real)

    array = array[2:]
    counter_array = 0
    counter = 0
    image_final = np.zeros((vertical, horizontal)) * 1j

    for i in range(horizontal):
        for j in range(vertical):
            if(counter == 0):
                if(array[counter_array] == 0):
                    counter = int((array[counter_array+1]-1).real)
                    counter_array += 2
                else:
                    image_final[j][i] = array[counter_array]
                    counter_array += 1
            else:
                counter -= 1

    return image_final


def noise_filter(f, k1, k2):
    if (k1>=1.0 or k2>=1.0):
        print("taux de filtre trop élevé\n")
    f_chapeau = tdf2(f)
    h = len(f_chapeau)
    v = len(f_chapeau[0])
    filterH = int((h*k1)/2)
    filterV = int((v*k2)/2)
    h_inf = int(h/2)-filterH
    h_sup = int(h/2)+filterH
    v_inf = int(v/2)-filterV
    v_sup = int(v/2)+filterV

    f_chapeau[h_inf : h_sup,] = 0
    f_chapeau[ : , v_inf : v_sup] = 0
    return itfd2(f_chapeau), f_chapeau

# for i in range(1,17) :
#     img = misc.imread("images/" + str(i) +".png")
#     img, tf = noise_filter(img, 0.8, 0.8)
#     misc.imsave(str(i) + ".png", np.abs(img))

img = misc.imread("cache.png")
img, tf = noise_filter(img, 0.8, 0.8)
misc.imsave("discover.png", np.abs(img))

img = misc.imread("transmetropolitan.pgm")

compression(img,0)
tf2 = decompression("image.david")
final = itfd2(tf2)
misc.imsave("final.pgm", np.abs(final))