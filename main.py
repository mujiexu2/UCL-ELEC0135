import resnet as resnet
import vgg as vgg
import time

def main():
    resnet.run()
    time.sleep(5)
    vgg.run()
    time.sleep(5)

if __name__ == '__main__':
    main()