from django.shortcuts import render
from django.shortcuts import redirect
from django import forms
from .forms import UploadImageForm, ImageUploadForm
from django.core.files.storage import FileSystemStorage
from django.conf import settings
#from pyp5js import *
import math
import numpy as np
import cv2
from .opencv_dface import opencv_dface
from django.http import HttpResponseRedirect

posx=[0]
posy=[0]
dir=["","",""]

def home(request):
  if request.method == 'GET':
    return render(request, 'opencv_webapp/uimage.html', {})
  if request.method == 'POST':
    return render(request, 'opencv_webapp/uimage.html', {'x':posx[0],'y':posy[0]})



def getPerspectiveTransform(sourcePoints, destinationPoints):
    a = np.zeros((8, 8))
    b = np.zeros((8))
    for i in range(4):
        a[i][0] = a[i+4][3] = sourcePoints[i][0]
        a[i][1] = a[i+4][4] = sourcePoints[i][1]
        a[i][2] = a[i+4][5] = 1
        a[i][3] = a[i][4] = a[i][5] = 0
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0
        a[i][6] = -sourcePoints[i][0]*destinationPoints[i][0]
        a[i][7] = -sourcePoints[i][1]*destinationPoints[i][0]
        a[i+4][6] = -sourcePoints[i][0]*destinationPoints[i][1]
        a[i+4][7] = -sourcePoints[i][1]*destinationPoints[i][1]
        b[i] = destinationPoints[i][0]
        b[i+4] = destinationPoints[i][1]
    x = np.linalg.solve(a, b)
    x.resize((9,), refcheck=False)
    x[8] = 1
    return x.reshape((3,3))

def warpPerspective(img,M,tam):
    rows,cols,sh=img.shape
    img2= np.zeros((tam[1],tam[0],3), np.uint8)
    for i in range(tam[1]):
        for j in range(tam[0]):
            I = np.linalg.inv(M)
            R = I.dot([[j],[i],[1]])
            R = R/R[2]
            homex=math.floor(R[0])
            homey=math.floor(R[1])
            img2.itemset((i,j,0),img.item(homey,homex,0))
            img2.itemset((i,j,1),img.item(homey,homex,1))
            img2.itemset((i,j,2),img.item(homey,homex,2))
    return img2

def puntos(request):
  if request.method == 'POST':
    if len(dir[0])!=0:
      img=cv2.imread(dir[0],cv2.IMREAD_GRAYSCALE)
      imgs=cv2.imread(dir[0])
      cols,rows,sh=imgs.shape
      H=10
      cv2.GaussianBlur(img, (11,11), 0, img)
      rows, cols = img.shape
      edges = cv2.Canny(img, 0, 100, apertureSize=3)
      (contornos,_) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      temp=[len(i) for i in contornos]
      pos_max=temp.index(max(temp))
      g=max(rows,cols)
      pts=[[g,g],[g,g],[0,0],[0,0]]
      for i in contornos[pos_max]:
        if pts[0][0]>i[0][0]:
          pts[0]=i[0]
        if pts[1][1]>i[0][1]:
          pts[1]=i[0]
        if pts[2][0]<i[0][0]:
          pts[2]=i[0]
        if pts[3][1]<i[0][1]:
          pts[3]=i[0]

      p1=punto_cercano([0,0],pts)
      p2=punto_cercano([cols,0],pts)
      p3=punto_cercano([0,rows],pts)
      p4=punto_cercano([cols,rows],pts)

      pts=[p1,p2,p3,p4]

      pxmin=min(pts[0][0],pts[1][0],pts[2][0],pts[3][0])
      pxmax=max(pts[0][0],pts[1][0],pts[2][0],pts[3][0])
      pymin=min(pts[0][1],pts[1][1],pts[2][1],pts[3][1])
      pymax=max(pts[0][1],pts[1][1],pts[2][1],pts[3][1])
      src=np.float32([pts[0],
              pts[1],
              pts[2],
              pts[3]])
      dst=np.float32([[0, 0],
              [pxmax-pxmin, 0],
              [0, pymax-pymin],
              [pxmax-pxmin, pymax-pymin]])
      matrix = cv2.getPerspectiveTransform(src, dst)
      cambiado = cv2.warpPerspective(imgs,matrix,(pxmax-pxmin,pymax-pymin))

      cambiado = contrast(cambiado)
      cv2.imwrite(dir[1],cambiado)
      cv2.imwrite(dir[2],cambiado)
      return HttpResponseRedirect('/uimage/')

def punto_cercano(p,pts):
    punto=[0,0]
    dist_min=np.inf
    k=0
    for i in pts:
        dist=math.sqrt(pow(p[0]-i[0],2)+pow(p[1]-i[1],2))
        if dist<dist_min:
            dist_min=dist
            punto=i
        k+=1
    return punto

def first_view(request):
  return HttpResponseRedirect('/uimage/')

def uimage(request):
  form = UploadImageForm(request.POST, request.FILES)
  if request.method == 'GET':
    if len(dir[0])==0:
      return render(request, 'opencv_webapp/uimage.html', {'form': form})
    else:
      return render(request, 'opencv_webapp/uimage.html', {'form': form, 'uploaded_file_url': dir[0][1:],'uploaded_file_url2': dir[1][1:]})

  elif request.method == 'POST':
    if form.is_valid():
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        temp=uploaded_file_url.split('.')
        dir[1]=settings.MEDIA_ROOT_URL+temp[0]+'1.'+temp[1]
        dir[0]=settings.MEDIA_ROOT_URL+uploaded_file_url
        dir[2]=settings.MEDIA_ROOT_URL+temp[0]+'2.'+temp[1]
        return render(request, 'opencv_webapp/uimage.html', {'form': form, 'uploaded_file_url': uploaded_file_url})
    else:
        form = UploadImageForm()
  return render(request, 'opencv_webapp/uimage.html',{'form':form})

def dface(request):
  form = ImageUploadForm(request.POST, request.FILES)
  if request.method == 'POST':
     if form.is_valid():
        post = form.save(commit=False)
        post.save()
        imageURL = settings.MEDIA_URL + form.instance.document.name
        dir[0]=settings.MEDIA_ROOT_URL + imageURL
        img=cv2.imread(dir[0], cv2.IMREAD_GRAYSCALE)
        cols, rows =img.shape
        cv2.imwrite(dir[0],img)
        print(cols,rows)
        return render(request, 'opencv_webapp/dface.html', {'form':form, 'post':post})
  else:
     form = ImageUploadForm()
  return render(request, 'opencv_webapp/dface.html',{'form':form})

def ajustar(request):
  if request.method == 'POST':
    if len(dir[0])!=0:
      px1=math.floor(float(request.POST['px1']))
      px2=math.floor(float(request.POST['px2']))
      px3=math.floor(float(request.POST['px3']))
      px4=math.floor(float(request.POST['px4']))
      py1=math.floor(float(request.POST['py1']))
      py2=math.floor(float(request.POST['py2']))
      py3=math.floor(float(request.POST['py3']))
      py4=math.floor(float(request.POST['py4']))
      img=cv2.imread(dir[0])
      cols,rows,sh=img.shape
      print(cols,rows)
      pts=np.float32([[px1, py1],
                [px2, py2],
                [px3, py3],
                [px4, py4]])
      print("pts: ",pts)
      p1=punto_cercano([0,0],pts)
      p2=punto_cercano([rows,0],pts)
      p3=punto_cercano([0,cols],pts)
      p4=punto_cercano([rows,cols],pts)
      print(p1,p2,p3,p4)
      pts=[p1,p2,p3,p4]
      pxmin=min(pts[0][0],pts[1][0],pts[2][0],pts[3][0])
      pxmax=max(pts[0][0],pts[1][0],pts[2][0],pts[3][0])
      pymin=min(pts[0][1],pts[1][1],pts[2][1],pts[3][1])
      pymax=max(pts[0][1],pts[1][1],pts[2][1],pts[3][1])

      print(pxmin,pxmax,pymin,pymax)
      src=np.float32([pts[0],
                    pts[1],
                    pts[2],
                    pts[3]])
      dst=np.float32([[0, 0],
                    [pxmax-pxmin, 0],
                    [0, pymax-pymin],
                    [pxmax-pxmin, pymax-pymin]])
      matrix = cv2.getPerspectiveTransform(src, dst)
     #matrix = getPerspectiveTransform(src, dst)
      cambiado = cv2.warpPerspective(img,matrix,(int(pxmax-pxmin),int(pymax-pymin)))
      #cambiado = warpPerspective(img,matrix,(pxmax-pxmin,pymax-pymin))
      cambiado = contrast(cambiado)
      cv2.imwrite(dir[1],cambiado)
      cv2.imwrite(dir[2],cambiado)
      return HttpResponseRedirect('/uimage/')

def contrast(img):
    img2 = img.copy()
    fils=len(img)
    cols=len(img[0])
    a=0
    b=255
    lista=[0]*256
    cont=fils*cols
    for x in range(fils):
        for y in range(cols):
            lista[img2.item(x,y,0)]=lista[img2.item(x,y,0)]+1
            lista[img2.item(x,y,1)]=lista[img2.item(x,y,1)]+1
            lista[img2.item(x,y,2)]=lista[img2.item(x,y,1)]+1
    lista2=[]
    for i in range(len(lista)):
        lista2=lista2+[i]*lista[i]
    x=5*cont/100
    y=95*cont/100

    c=0
    d=lista2[len(lista2)-1]

    print(x,y,c,d)
    for x in range(fils):
        for y in range(cols):
            t=(img.item(x,y,0)-c)*((b-a)/(d-c))+a
            if t<0:
                t=0
            if t>255:
                t=255
            img2.itemset((x, y,0), t)
            t=(img.item(x,y,1)-c)*((b-a)/(d-c))+a
            if t<0:
                t=0
            if t>255:
                t=255
            img2.itemset((x, y,1), t)
            t=(img.item(x,y,2)-c)*((b-a)/(d-c))+a
            if t<0:
                t=0
            if t>255:
                t=255
            img2.itemset((x, y,2), t)
    img = img2
    return img

def thre(request):
    if request.method == 'POST':
        if len(dir[2])!=0:

            img=cv2.imread(dir[2])
            Rxval=110
            Gxval=110
            Bxval=110
            fils = len(img)
            cols = len(img[0])
            bolin = 0
            for x in range(fils):
                for y in range(cols):
                    #RED
                    if img.item(x,y,0)<=Rxval:
                        bolin = bolin+1
                    #GREEN
                    if img.item(x,y,1)<=Gxval:
                        bolin = bolin+1
                    #BLUE
                    if img.item(x,y,2)<=Bxval:
                        bolin = bolin+1

                    if bolin == 3:
                        img.itemset((x,y,0),0)
                        img.itemset((x,y,1),0)
                        img.itemset((x,y,2),0)
                    else:
                        img.itemset((x,y,0),255)
                        img.itemset((x,y,1),255)
                        img.itemset((x,y,2),255)
                    bolin = 0

            cv2.imwrite(dir[1],img)
            return HttpResponseRedirect('/uimage/')

def gris(request):
    if request.method == 'POST':
        if len(dir[2])!=0:
            img=cv2.imread(dir[2],cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(dir[1],img)
            return HttpResponseRedirect('/uimage/')

def Rotate(angulo,img):
    rows,cols, ch = img.shape
    c=math.sqrt(pow(rows,2)+pow(cols,2))
    ang_img=math.acos(-(pow(cols,2)-pow(rows,2)-pow(c,2))/(2*rows*c))
    print(ang_img)
    if math.pi/2<=(angulo%(2*math.pi))<math.pi:
        X=-c*math.sin(-angulo+ang_img)
        Y=-c*math.sin(math.pi/2+angulo+ang_img)
    elif math.pi<=(angulo%(2*math.pi))<math.pi*3/2:
        X=-c*math.sin(angulo+ang_img)
        Y=-c*math.sin(math.pi/2-angulo+ang_img)
    elif math.pi*3/2<=(angulo%(2*math.pi))<2*math.pi:
        X=c*math.sin(-angulo+ang_img)
        Y=c*math.sin(math.pi/2+angulo+ang_img)
    else:
        X=c*math.sin(angulo+ang_img)
        Y=c*math.sin(math.pi/2-angulo+ang_img)

    X1=math.ceil(X)
    X2=math.floor(X)
    Y1=math.ceil(Y)
    Y2=math.floor(Y)
    N=np.array([[1.,   0.,  cols],
                [0.,   1.,  rows]])
    img2 = cv2.warpAffine(img, N, (cols*3,rows*3))
    rows1,cols1, ch1 = img2.shape
    N=np.array([[math.cos(angulo)   ,   math.sin(angulo)    ,  (1-math.cos(angulo))*cols1/2   -   math.sin(angulo)      *rows1/2],
                [-math.sin(angulo)  ,   math.cos(angulo)    ,   math.sin(angulo)   *cols1/2   +   (1-math.cos(angulo))  *rows1/2]])
    img3 = cv2.warpAffine(img2, N, (cols*3,rows*3))
    q=(X1-cols)/2
    w=(Y1-rows)/2
    print(q,w)
    N=np.array([[1.,   0.,  -cols+q],
                [0.,   1.,  -rows+w]])
    print(X,Y)
    print((abs(X1),abs(Y1)))
    img4 = cv2.warpAffine(img3, N, (abs(X1),abs(Y1)))
    return img4

def giro(request):
    if request.method == 'POST':
        if len(dir[1])!=0:
            img=cv2.imread(dir[1])
            img = Rotate(math.pi/2,img)
            cv2.imwrite(dir[1],img)
            cv2.imwrite(dir[2],img)
            return HttpResponseRedirect('/uimage/')

def color(request):
    if request.method == 'POST':
        if len(dir[2])!=0:
            img=cv2.imread(dir[2])
            cv2.imwrite(dir[1],img)
            cv2.imwrite(dir[2],img)
            return HttpResponseRedirect('/uimage/')

def Adaptative(request):
    if request.method == 'POST':
        if len(dir[2])!=0:
            img=cv2.imread(dir[2])
            imagen = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh2 = cv2.adaptiveThreshold(imagen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 21, 4)
            cv2.imwrite(dir[1],thresh2)
            return HttpResponseRedirect('/uimage/#menu1')
