//
//  Rotate+UIImage.swift
//  AVCamBarcode
//
//  Created by Tony on 24.06.17.
//  Copyright © 2017 Apple, Inc. All rights reserved.
//

import Foundation
import UIKit
import AVFoundation

extension UIImage {
    
    func degreesToRadians(_ degrees:Double) -> CGFloat {
        return CGFloat(degrees) * CGFloat(Double.pi) / CGFloat(180.0)
    }
    
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
    
    func resize(to targetSize: CGSize) -> UIImage {
        let size = self.size
        
        let widthRatio  = targetSize.width  / self.size.width
        let heightRatio = targetSize.height / self.size.height
        
        // Figure out what our orientation is, and use that to form the rectangle
        var newSize: CGSize
        if(widthRatio > heightRatio) {
            newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
        } else {
            newSize = CGSize(width: size.width * widthRatio,  height: size.height * widthRatio)
        }
        
        // This is the rect that we've calculated out and this is what is actually used below
        let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)
        
        // Actually do the resizing to the rect using the ImageContext stuff
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        self.draw(in: rect)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage!
    }
}


// Image Utils
let neuralInputSize = CGSize(width: 48, height: 48)
public func cropToPreviewLayer(originalImage: CGImage, layer: AVCaptureVideoPreviewLayer, faceBounds: CGRect) -> UIImage {
    let outputRect = layer.metadataOutputRectOfInterest(for: faceBounds)
    var cgImage = originalImage
    let width = CGFloat(cgImage.width)
    let height = CGFloat(cgImage.height)
    let cropRect = CGRect(x: outputRect.origin.x * width, y: outputRect.origin.y * height, width: outputRect.size.width * width, height: outputRect.size.height * height)
    
    cgImage = cgImage.cropping(to: cropRect)!
    
    // The tonal is a bit lighter
    let currentFilter = CIFilter(name: "CIPhotoEffectTonal") //CIPhotoEffectNoir
    currentFilter!.setValue(CIImage(cgImage: cgImage), forKey: kCIInputImageKey)
    let output = currentFilter!.outputImage
    let context = CIContext(options: nil)
    let grayScale = context.createCGImage(output!,from: output!.extent)
    
    let croppedUIImage = UIImage(cgImage: grayScale ?? cgImage, scale: 1.0, orientation: .downMirrored).rotate(byDegrees: 90).resize(to: neuralInputSize)
    //        UIImageWriteToSavedPhotosAlbum(croppedUIImage, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
    
    return croppedUIImage
}

public func pixelBufferGray(from image: UIImage) -> CVPixelBuffer? {
    
    let width = Int(image.size.width)
    let height = Int(image.size.height)
    
    var pixelBuffer : CVPixelBuffer?
    let attributes = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
    
    let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(width), Int(height), kCVPixelFormatType_OneComponent8, attributes as CFDictionary, &pixelBuffer)
    
    guard status == kCVReturnSuccess, let imageBuffer = pixelBuffer else {
        return nil
    }
    
    CVPixelBufferLockBaseAddress(imageBuffer, CVPixelBufferLockFlags(rawValue: 0))
    
    let imageData =  CVPixelBufferGetBaseAddress(imageBuffer)
    
    guard let context = CGContext(data: imageData, width: Int(width), height:Int(height),
                                  bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(imageBuffer),
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
                                    return nil
    }
    
    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: 1, y: -1)
    
    UIGraphicsPushContext(context)
    image.draw(in: CGRect(x:0, y:0, width: width, height: height) )
    UIGraphicsPopContext()
    CVPixelBufferUnlockBaseAddress(imageBuffer, CVPixelBufferLockFlags(rawValue: 0))
    
    return imageBuffer
    
}



