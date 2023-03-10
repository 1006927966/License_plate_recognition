import  recognition.crnn as crnn

def getrecogmodel():
    return  crnn.CRNN(32, 1, 68, 256)