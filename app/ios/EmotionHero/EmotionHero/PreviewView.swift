/*
    Copyright (C) 2016 Apple Inc. All Rights Reserved.
    See LICENSE.txt for this sample’s licensing information
	
    Abstract:
    Application preview view.
*/

import UIKit
import AVFoundation

class PreviewView: UIView {
    // MARK: Properties
    
    private let regionOfInterestCornerTouchThreshold: CGFloat = 50
    private let maskLayer = CAShapeLayer()
    private let regionOfInterestOutline = CAShapeLayer()
    // Change the ROI size here
    //private static let phoneScreen: CGRect = UIScreen.main.bounds.standardized
    //private(set) var regionOfInterest = CGRect(x:0, y:0, width:phoneScreen.width, height: viewHeight)
    private var regionOfInterest: CGRect!
    
    // MARK: AV capture properties
    
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        return layer as! AVCaptureVideoPreviewLayer
    }
    
    var session: AVCaptureSession? {
        get {
            return videoPreviewLayer.session
        }
        
        set{
            videoPreviewLayer.session = newValue
        }
    }
    
    // MARK: Initialization
	
    override init(frame: CGRect) {
        super.init(frame: frame)
        
        commonInit()
    }
	
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
        
        commonInit()
    }
	
    private func commonInit() {
        maskLayer.fillRule = kCAFillRuleEvenOdd
        maskLayer.fillColor = UIColor.black.cgColor
        maskLayer.opacity = 0.6
        regionOfInterest = CGRect(x:0, y:0, width: bounds.width, height: bounds.height)
        layer.addSublayer(maskLayer)
        
        regionOfInterestOutline.path = UIBezierPath(rect: regionOfInterest).cgPath
        regionOfInterestOutline.fillColor = UIColor.clear.cgColor
        regionOfInterestOutline.strokeColor = UIColor.yellow.cgColor
        layer.addSublayer(regionOfInterestOutline)
    }
    
    // MARK: UIView
	
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }
	
    override func layoutSubviews() {
        super.layoutSubviews()
        
        // Disable CoreAnimation actions so that the positions of the sublayers immediately move to their new position.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        
        // Create the path for the mask layer. We use the even odd fill rule so that the region of interest does not have a fill color.
        let path = UIBezierPath(rect: CGRect(x: 0, y: 0, width: frame.size.width, height: frame.size.height))
        path.append(UIBezierPath(rect: regionOfInterest))
        path.usesEvenOddFillRule = true
        maskLayer.path = path.cgPath
        
        regionOfInterestOutline.path = CGPath(rect: regionOfInterest, transform: nil)
        
        CATransaction.commit()
    }
}
