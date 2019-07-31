'''Example parameters: 400000 500000 0 100000

Example with Empty result set
500000 600000 -100000 0

200000 300000 800000 900000

'''
import argparse
import MtnGlaGridcellProcess as mtn



def parseArguments():
    parser = argparse.ArgumentParser(description='Clip and publish mask of an extent.')
    parser.add_argument('minX', type=int, help='minX of extent')
    parser.add_argument('maxX', type=int, help='maxX of extent')
    parser.add_argument('minY', type=int, help='minY of extent')
    parser.add_argument('maxY', type=int, help='maxY of extent')

    return parser.parse_args()


if __name__=="__main__":
    args=parseArguments()
    process = mtn.MtnGlaGridcellProcess(args.minX, args.maxX, args.minY, args.maxY, logFile='mntgla.log')
    process.startProcess()



