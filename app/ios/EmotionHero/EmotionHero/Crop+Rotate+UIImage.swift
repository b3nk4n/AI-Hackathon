//
//  Rotate+UIImage.swift
//  AVCamBarcode
//
//  Created by Tony on 24.06.17.
//  Copyright © 2017 Apple, Inc. All rights reserved.
//

import Foundation
import UIKit

extension UIImage {
    
    func rotate(byDegrees degree: Double) -> UIImage {
        let radians = degreesToRadians(degree)
        let rotatedSize = self.size
        let scale = UIScreen.main.scale
        UIGraphicsBeginImageContextWithOptions(rotatedSize, false, scale)
        let bitmap = UIGraphicsGetCurrentContext()
        bitmap?.translateBy(x:rotatedSize.width / 2, y:rotatedSize.height / 2)
        bitmap?.rotate(by: radians)
        bitmap?.translateBy(x:1.0, y:-1.0);
        bitmap?.draw(self.cgImage!, in: CGRect(x: -self.size.width/2, y: -self.size.height/2, width: self.size.width, height: self.size.height))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage!
    }
    
    func crop( rect: CGRect) -> UIImage {
        var rect = rect
        rect.origin.x*=self.scale
        rect.origin.y*=self.scale
        rect.size.width*=self.scale
        rect.size.height*=self.scale
        
        let imageRef = self.cgImage!.cropping(to: rect)
        let image = UIImage(cgImage: imageRef!, scale: self.scale, orientation: self.imageOrientation)
        return image
    }
}


let kOrientationToDegreesFront: [UIDeviceOrientation: CGFloat] = [
    .portrait: -90,
    .portraitUpsideDown: 90,
    .landscapeLeft: 180,
    .landscapeRight: 0,
    .faceUp: 0,
    .faceDown: 0
]

let kOrientationToDegreesBack: [UIDeviceOrientation: CGFloat] = [
    .portrait: -90,
    .portraitUpsideDown: 90,
    .landscapeLeft: 0,
    .landscapeRight: 180,
    .faceUp: 0,
    .faceDown: 0
]


// Image Utils

/* kCGImagePropertyOrientation values
 The intended display orientation of the image. If present, this key is a CFNumber value with the same value as defined
 by the TIFF and EXIF specifications -- see enumeration of integer constants.
 The value specified where the origin (0,0) of the image is located. If not present, a value of 1 is assumed.
 
 used when calling featuresInImage: options: The value for this key is an integer NSNumber from 1..8 as found in kCGImagePropertyOrientation.
 If present, the detection will be done based on that orientation but the coordinates in the returned features will still be based on those of the image. */

enum PhotosExif0Row: Int {
    case TOP_0COL_LEFT            = 1 //   1  =  0th row is at the top, and 0th column is on the left (THE DEFAULT).
    case TOP_0COL_RIGHT            = 2 //   2  =  0th row is at the top, and 0th column is on the right.
    case BOTTOM_0COL_RIGHT      = 3 //   3  =  0th row is at the bottom, and 0th column is on the right.
    case BOTTOM_0COL_LEFT       = 4 //   4  =  0th row is at the bottom, and 0th column is on the left.
    case LEFT_0COL_TOP          = 5 //   5  =  0th row is on the left, and 0th column is the top.
    case RIGHT_0COL_TOP         = 6 //   6  =  0th row is on the right, and 0th column is the top.
    case RIGHT_0COL_BOTTOM      = 7 //   7  =  0th row is on the right, and 0th column is the bottom.
    case LEFT_0COL_BOTTOM       = 8  //   8  =  0th row is on the left, and 0th column is the bottom.
}

let kDeviceOrientationToExifOrientationFront: [UIDeviceOrientation: PhotosExif0Row] = [
    .portrait: .RIGHT_0COL_TOP,
    .portraitUpsideDown: .LEFT_0COL_BOTTOM,
    .landscapeLeft: .BOTTOM_0COL_RIGHT,
    .landscapeRight: .TOP_0COL_LEFT
]

let kDeviceOrientationToExifOrientationBack: [UIDeviceOrientation: PhotosExif0Row] = [
    .portrait: .RIGHT_0COL_TOP,
    .portraitUpsideDown: .LEFT_0COL_BOTTOM,
    .landscapeLeft: .TOP_0COL_LEFT,
    .landscapeRight: .BOTTOM_0COL_RIGHT
]

//  Maps a Bool, representing whether the front facing camera is being used, to the correct
//  dictionary that itself maps the device orientation to the correcnt EXIF orientation.
let kDeviceOrientationToExifOrientation: [Bool: [UIDeviceOrientation: PhotosExif0Row]] = [
    true: kDeviceOrientationToExifOrientationFront,
    false: kDeviceOrientationToExifOrientationBack
]

func degreesToRadians(_ degrees:Double) -> CGFloat {
    return CGFloat(degrees) * CGFloat(Double.pi) / CGFloat(180.0)
}

func RotationTransform(degrees:Float) -> CGAffineTransform
{
    return CGAffineTransform(rotationAngle: degreesToRadians(Double(degrees)))
}

func newSquareOverlayedImageForFeatures (
    squareImage: UIImage,
    //features: [CIFaceFeature],
    faceRect: CGRect,
    backgroundImage: CGImage,
    orientation: UIDeviceOrientation,
    isFrontFacing: Bool) -> CGImage
{
    var returnImage: CGImage!
    let w  = Int(backgroundImage.width)
    let h  = Int(backgroundImage.height)
    let backgroundImageRect = CGRect(x:0, y:0, width: w, height:h)
    
    var bitmapContext: CGContext! = createCGBitmapContextFor(backgroundImageRect.size)
    bitmapContext.clear(backgroundImageRect)
    bitmapContext.draw(backgroundImage, in: backgroundImageRect)
    //  Use dictionaries to look up the rotation corresponding to the given orientation
    if let rotationDegrees = isFrontFacing ?
        kOrientationToDegreesFront[orientation] : kOrientationToDegreesBack[orientation] {
        
        let rotatedSquareImage = squareImage.rotate(byDegrees: Double(rotationDegrees))
        
        // features found by the face detector
//        for ff in features {
//            let faceRect = ff.bounds
            bitmapContext.draw(rotatedSquareImage.cgImage!, in: faceRect)
//        }
        returnImage = bitmapContext.makeImage()
        
    }
    return returnImage;
}

func createCGBitmapContextFor(_ size: CGSize) -> CGContext
{
    let colorSpace:CGColorSpace! = CGColorSpaceCreateDeviceRGB();
    let bytesPerRow = size.width * 4
    let bitsPerComponent = 8
    
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    let context = CGContext(data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: bitsPerComponent, bytesPerRow: Int(bytesPerRow), space: colorSpace, bitmapInfo: bitmapInfo.rawValue)
    
    context!.setAllowsAntialiasing(false);
    return context!
}
