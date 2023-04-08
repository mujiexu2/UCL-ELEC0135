import resnet as resnet
import vgg as vgg
import time

def main():
    print("-----------------------Running: Resnet-18 ---------------------------------------")
    resnet.run()
    time.sleep(5)
    print("-----------------------Running: VGG-19 ------------------------------------------")
    vgg.run()
    time.sleep(5)

if __name__ == '__main__':
    main()