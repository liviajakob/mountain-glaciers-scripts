'''Example parameters: 400000 500000 0 100000

'''
import argparse
import ClipMaskProcess as clip



def parseArguments():
    parser = argparse.ArgumentParser(description='Clip and publish mask of an extent.')
    parser.add_argument('minX', type=int, help='minX of extent')
    parser.add_argument('maxX', type=int, help='maxX of extent')
    parser.add_argument('minY', type=int, help='minY of extent')
    parser.add_argument('maxY', type=int, help='maxY of extent')

    return parser.parse_args()


if __name__=="__main__":
    args=parseArguments()
    process = clip.ClipMaskProcess(args.minX, args.maxX, args.minY, args.maxY, logFile='clip.log')
    process.startProcess()



