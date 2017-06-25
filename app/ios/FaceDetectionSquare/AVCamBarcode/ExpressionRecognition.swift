//
//  ExpressionRecognition.swift
//  AVCamBarcode
//
//  Created by Tony on 25.06.17.
//  Copyright © 2017 Apple, Inc. All rights reserved.
//

import UIKit
import CoreML

extension MLMultiArray {
    var array: [Double] {
        get{
            var temp = [Double]()
            for i in 0..<self.count {
                if let val = self[i] as? Double{
                    temp.append(val)
                } else {
                    temp.append(0)
                }
            }
            return temp
        }
    }
    
    // Max value and its index
    var maxId: (max:Double?, index: Int?) {
        get {
            return (self.array.max(), self.array.index(of: self.array.max() ?? 0.0))
        }
    }
    
    // Output label 
}


// MARK: Emotion prediction
let neuralModel = keras_model()
public let emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
enum Emotions: Int {
    case angry
    case disgust
    case fear
    case happy
    case neutral
    case sad
    case surprise
}
public func predictEmotion(faceImage: UIImage) -> String {
    let convertedImage = pixelBufferGray(from: faceImage)
    if let buffer = convertedImage {
        guard let prediction = try? neuralModel.prediction(input_1: buffer) else {
            print("error")
            return "neutral"
        }
        print("emotions: \(emotions)")
        print("prediction: \(prediction.output1)")
        print("pred array: \(prediction.output1.maxId)")
        print("emotions: \(emotions[prediction.output1.maxId.index ?? 0]) with prob \(prediction.output1.maxId.max)")
        
        return emotions[prediction.output1.maxId.index ?? 4] // Return emotion, if nil return neutral
    } else {
        return "neutral"
    }
}
