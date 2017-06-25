/*
 Copyright (C) 2016 Apple Inc. All Rights Reserved.
 See LICENSE.txt for this sample’s licensing information
 
 Abstract:
 View controller for camera interface.
 */

import UIKit
import AVFoundation
import CoreML

class CameraViewController: UIViewController, AVCaptureMetadataOutputObjectsDelegate, AVCapturePhotoCaptureDelegate, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    // MARK: View Controller Life Cycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Set up the video preview view.
        session.sessionPreset = AVCaptureSessionPreset640x480
        previewView.session = session
        
        
        /*
         Check video authorization status. Video access is required and audio
         access is optional. If audio access is denied, audio is not recorded
         during movie recording.
         */
        switch AVCaptureDevice.authorizationStatus(forMediaType: AVMediaTypeVideo) {
        case .authorized:
            // The user has previously granted access to the camera.
            break
            
        case .notDetermined:
            /*
             The user has not yet been presented with the option to grant
             video access. We suspend the session queue to delay session
             setup until the access request has completed.
             */
            sessionQueue.suspend()
            AVCaptureDevice.requestAccess(forMediaType: AVMediaTypeVideo, completionHandler: { [unowned self] granted in
                if !granted {
                    self.setupResult = .notAuthorized
                }
                self.sessionQueue.resume()
            })
            
        default:
            // The user has previously denied access.
            setupResult = .notAuthorized
        }
        
        /*
         Setup the capture session.
         In general it is not safe to mutate an AVCaptureSession or any of its
         inputs, outputs, or connections from multiple threads at the same time.
         
         Why not do all of this on the main queue?
         Because AVCaptureSession.startRunning() is a blocking call which can
         take a long time. We dispatch session setup to the sessionQueue so
         that the main queue isn't blocked, which keeps the UI responsive.
         */
        sessionQueue.async { [unowned self] in
            self.configureSession()
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        sessionQueue.async { [unowned self] in
            switch self.setupResult {
            case .success:
                // Only setup observers and start the session running if setup succeeded.
                self.addObservers()
                self.session.startRunning()
                self.isSessionRunning = self.session.isRunning
                
            case .notAuthorized:
                DispatchQueue.main.async { [unowned self] in
                    let message = NSLocalizedString("AVCamBarcode doesn't have permission to use the camera, please change privacy settings", comment: "Alert message when the user has denied access to the camera")
                    let    alertController = UIAlertController(title: "AVCamBarcode", message: message, preferredStyle: .alert)
                    alertController.addAction(UIAlertAction(title: NSLocalizedString("OK", comment: "Alert OK button"), style: .cancel, handler: nil))
                    alertController.addAction(UIAlertAction(title: NSLocalizedString("Settings", comment: "Alert button to open Settings"), style: .`default`, handler: { action in
                        UIApplication.shared.open(URL(string: UIApplicationOpenSettingsURLString)!, options: [:], completionHandler: nil)
                    }))
                    
                    self.present(alertController, animated: true, completion: nil)
                }
                
            case .configurationFailed:
                DispatchQueue.main.async { [unowned self] in
                    let message = NSLocalizedString("Unable to capture media", comment: "Alert message when something goes wrong during capture session configuration")
                    let alertController = UIAlertController(title: "AVCamBarcode", message: message, preferredStyle: .alert)
                    alertController.addAction(UIAlertAction(title: NSLocalizedString("OK", comment: "Alert OK button"), style: .cancel, handler: nil))
                    
                    self.present(alertController, animated: true, completion: nil)
                }
            }
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        sessionQueue.async { [unowned self] in
            if self.setupResult == .success {
                self.session.stopRunning()
                self.isSessionRunning = self.session.isRunning
                self.removeObservers()
            }
        }
        
        super.viewWillDisappear(animated)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // For later if we have any transitions
    }
    
    override var shouldAutorotate: Bool {
        // Do not allow rotation if the region of interest is being resized.
        return false //!previewView.isResizingRegionOfInterest
    }
    
    // MARK: Session Management
    
    private enum SessionSetupResult {
        case success
        case notAuthorized
        case configurationFailed
    }
    
    private let session = AVCaptureSession()
    
    private var isSessionRunning = false
    
    private let sessionQueue = DispatchQueue(label: "session queue", attributes: [], target: nil) // Communicate with the session and other session objects on this queue.
    
    private var setupResult: SessionSetupResult = .success
    
    var videoDeviceInput: AVCaptureDeviceInput!
    
    @IBOutlet private var previewView: PreviewView!
    
    // Call this on the session queue.
    private func configureSession() {
        if self.setupResult != .success {
            return
        }
        
        session.beginConfiguration()
        
        // Add video input.
        do {
            let frontCameraDevice = AVCaptureDevice.defaultDevice(withDeviceType: .builtInWideAngleCamera, mediaType: AVMediaTypeVideo, position: .front)
            
            let videoDeviceInput = try AVCaptureDeviceInput(device: frontCameraDevice)
            
            if session.canAddInput(videoDeviceInput) {
                session.addInput(videoDeviceInput)
                self.videoDeviceInput = videoDeviceInput
                
                DispatchQueue.main.async {
                    /*
                     Why are we dispatching this to the main queue?
                     Because AVCaptureVideoPreviewLayer is the backing layer for PreviewView and UIView
                     can only be manipulated on the main thread.
                     Note: As an exception to the above rule, it is not necessary to serialize video orientation changes
                     on the AVCaptureVideoPreviewLayer’s connection with other session manipulation.
                     
                     Use the status bar orientation as the initial video orientation. Subsequent orientation changes are
                     handled by CameraViewController.viewWillTransition(to:with:).
                     */
                    let statusBarOrientation = UIApplication.shared.statusBarOrientation
                    var initialVideoOrientation: AVCaptureVideoOrientation = .portrait
                    if statusBarOrientation != .unknown {
                        if let videoOrientation = statusBarOrientation.videoOrientation {
                            initialVideoOrientation = videoOrientation
                        }
                    }
                    
                    self.previewView.videoPreviewLayer.connection.videoOrientation = initialVideoOrientation
                }
            }
            else {
                print("Could not add video device input to the session")
                setupResult = .configurationFailed
                session.commitConfiguration()
                return
            }
        }
        catch {
            print("Could not create video device input: \(error)")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        // Add metadata output.
        if session.canAddOutput(metadataOutput) {
            session.addOutput(metadataOutput)
            
            // Set this view controller as the delegate for metadata objects.
            metadataOutput.setMetadataObjectsDelegate(self, queue: metadataObjectsQueue)
            metadataOutput.metadataObjectTypes = metadataOutput.availableMetadataObjectTypes // Use all metadata object types by default.
        } else {
            print("Could not add metadata output to the session")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        // Add still capture
        if session.canAddOutput(stillImageOutput) {
            session.addOutput(stillImageOutput)
        }
        session.commitConfiguration()
    }
    
    private let metadataOutput = AVCaptureMetadataOutput()
    
    private let metadataObjectsQueue = DispatchQueue(label: "metadata objects queue", attributes: [], target: nil)
    
    // MARK: Device Configuration
    
    
    @IBOutlet private var cameraUnavailableLabel: UILabel!
    
    private let videoDeviceDiscoverySession = AVCaptureDeviceDiscoverySession(deviceTypes: [.builtInWideAngleCamera, .builtInDuoCamera], mediaType: AVMediaTypeVideo, position: .unspecified)!
    
    // MARK: KVO and Notifications
    
    private var sessionRunningObserveContext = 0
    
    private func addObservers() {
        session.addObserver(self, forKeyPath: "running", options: .new, context: &sessionRunningObserveContext)
        
        NotificationCenter.default.addObserver(self, selector: #selector(sessionRuntimeError), name: Notification.Name("AVCaptureSessionRuntimeErrorNotification"), object: session)
        
        /*
         A session can only run when the app is full screen. It will be interrupted
         in a multi-app layout, introduced in iOS 9, see also the documentation of
         AVCaptureSessionInterruptionReason. Add observers to handle these session
         interruptions and show a preview is paused message. See the documentation
         of AVCaptureSessionWasInterruptedNotification for other interruption reasons.
         */
        NotificationCenter.default.addObserver(self, selector: #selector(sessionWasInterrupted), name: Notification.Name("AVCaptureSessionWasInterruptedNotification"), object: session)
        NotificationCenter.default.addObserver(self, selector: #selector(sessionInterruptionEnded), name: Notification.Name("AVCaptureSessionInterruptionEndedNotification"), object: session)
    }
    
    private func removeObservers() {
        NotificationCenter.default.removeObserver(self)
        
        session.removeObserver(self, forKeyPath: "running", context: &sessionRunningObserveContext)
    }
    
    override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey : Any]?, context: UnsafeMutableRawPointer?) {
        let newValue = change?[.newKey] as AnyObject?
        
        if context == &sessionRunningObserveContext {
            guard let isSessionRunning = newValue?.boolValue else { return }
            
            DispatchQueue.main.async { [unowned self] in
                /*
                 After the session stop running, remove the metadata object overlays,
                 if any, so that if the view appears again, the previously displayed
                 metadata object overlays are removed.
                 */
                if !isSessionRunning {
                    self.removeMetadataObjectOverlayLayers()
                }
            }
        }
        else {
            super.observeValue(forKeyPath: keyPath, of: object, change: change, context: context)
        }
    }
    
    func sessionRuntimeError(notification: NSNotification) {
        guard let errorValue = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError else { return }
        
        let error = AVError(_nsError: errorValue)
        print("Capture session runtime error: \(error)")
        
        /*
         Automatically try to restart the session running if media services were
         reset and the last start running succeeded. Otherwise, enable the user
         to try to resume the session running.
         */
        if error.code == .mediaServicesWereReset {
            sessionQueue.async { [unowned self] in
                if self.isSessionRunning {
                    self.session.startRunning()
                    self.isSessionRunning = self.session.isRunning
                }
            }
        }
    }
    
    func sessionWasInterrupted(notification: NSNotification) {
        /*
         In some scenarios we want to enable the user to resume the session running.
         For example, if music playback is initiated via control center while
         using AVCamBarcode, then the user can let AVCamBarcode resume
         the session running, which will stop music playback. Note that stopping
         music playback in control center will not automatically resume the session
         running. Also note that it is not always possible to resume, see `resumeInterruptedSession(_:)`.
         */
        if let userInfoValue = notification.userInfo?[AVCaptureSessionInterruptionReasonKey] as AnyObject?, let reasonIntegerValue = userInfoValue.integerValue, let reason = AVCaptureSessionInterruptionReason(rawValue: reasonIntegerValue) {
            print("Capture session was interrupted with reason \(reason)")
            
            if reason == AVCaptureSessionInterruptionReason.videoDeviceNotAvailableWithMultipleForegroundApps {
                // Simply fade-in a label to inform the user that the camera is unavailable.
                self.cameraUnavailableLabel.isHidden = false
                self.cameraUnavailableLabel.alpha = 0
                UIView.animate(withDuration: 0.25) {
                    self.cameraUnavailableLabel.alpha = 1
                }
            }
        }
    }
    
    func sessionInterruptionEnded(notification: NSNotification) {
        print("Capture session interruption ended")
        
        if cameraUnavailableLabel.isHidden {
            UIView.animate(withDuration: 0.25,
                           animations: { [unowned self] in
                            self.cameraUnavailableLabel.alpha = 0
                }, completion: { [unowned self] finished in
                    self.cameraUnavailableLabel.isHidden = true
                }
            )
        }
    }
    
    // MARK: Drawing Metadata Object Overlay Layers
    
    private class MetadataObjectLayer: CAShapeLayer {
        var metadataObject: AVMetadataObject?
    }
    
    /**
     A dispatch semaphore is used for drawing metadata object overlays so that
     only one group of metadata object overlays is drawn at a time.
     */
    private let metadataObjectsOverlayLayersDrawingSemaphore = DispatchSemaphore(value: 1)
    
    private var metadataObjectOverlayLayers = [MetadataObjectLayer]()
    
    // Create the initial metadata object overlay layer that can be used for either machine readable codes or faces.
    private let metadataObjectOverlayLayer = MetadataObjectLayer()
    private func createMetadataObjectOverlayWithMetadataObject(_ metadataObject: AVMetadataObject) -> MetadataObjectLayer {
        // Transform the metadata object so the bounds are updated to reflect those of the video preview layer.
        let transformedMetadataObject = previewView.videoPreviewLayer.transformedMetadataObject(for: metadataObject)
        let faceRect = transformedMetadataObject!.bounds
        
        // Increace face boxing by pct %
        let pct: CGFloat = 1.2; // increase in %: 1.2 -> +20%, 0.5 -> -50%
        let width = faceRect.width
        let height = faceRect.height
        let newWidth = sqrt(width * width * pct)
        let newHeight = sqrt(height * height * pct)
        let newRect = faceRect.insetBy(dx:(width-newWidth)/2, dy:(height-newHeight)/2);
        
        // Only detect faces
        if transformedMetadataObject is AVMetadataFaceObject {
            print("face")
            metadataObjectOverlayLayer.metadataObject = transformedMetadataObject
            metadataObjectOverlayLayer.lineJoin = kCALineJoinRound
            metadataObjectOverlayLayer.lineWidth = 7.0
            metadataObjectOverlayLayer.strokeColor = view.tintColor.withAlphaComponent(0.7).cgColor
            metadataObjectOverlayLayer.fillColor = view.tintColor.withAlphaComponent(0.3).cgColor
            metadataObjectOverlayLayer.path = CGPath(rect: newRect, transform: nil)
            
            // Save face rect
            faceBounds = newRect
            
        } else {
            print("no face")
        }
        
        return metadataObjectOverlayLayer
    }
    
    private var removeMetadataObjectOverlayLayersTimer: Timer?
    
    @objc private func removeMetadataObjectOverlayLayers() {
        for sublayer in metadataObjectOverlayLayers {
            sublayer.removeFromSuperlayer()
        }
        metadataObjectOverlayLayers = []
        removeMetadataObjectOverlayLayersTimer?.invalidate()
        removeMetadataObjectOverlayLayersTimer = nil
    }
    
    private func addMetadataObjectOverlayLayersToVideoPreviewView(_ metadataObjectOverlayLayers: [MetadataObjectLayer]) {
        // Add the metadata object overlays as sublayers of the video preview layer. We disable actions to allow for fast drawing.
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        for metadataObjectOverlayLayer in metadataObjectOverlayLayers {
            previewView.videoPreviewLayer.addSublayer(metadataObjectOverlayLayer)
        }
        CATransaction.commit()
        
        // Save the new metadata object overlays.
        self.metadataObjectOverlayLayers = metadataObjectOverlayLayers
        
        // Create a timer to destroy the metadata object overlays.
        removeMetadataObjectOverlayLayersTimer = Timer.scheduledTimer(timeInterval: 1, target: self, selector: #selector(removeMetadataObjectOverlayLayers), userInfo: nil, repeats: false)
    }
    
    // MARK: Capture still image
    
    let stillImageOutput = AVCapturePhotoOutput()
    var faceBounds = CGRect.null
    @IBOutlet weak var capturedImage: UIImageView!
    public var faceImage = UIImage()
    
    // Take picture button
    @IBAction func didPressTakePhoto(_ sender: UIButton) {
        captureFace()
    }
    
    func captureFace() {
        let settings = AVCapturePhotoSettings()
        let previewPixelType = settings.availablePreviewPhotoPixelFormatTypes.first!
        let previewFormat = [
            kCVPixelBufferPixelFormatTypeKey as String: previewPixelType,
            kCVPixelBufferWidthKey as String: 160,
            kCVPixelBufferHeightKey as String: 160
        ]
        settings.previewPhotoFormat = previewFormat
        stillImageOutput.capturePhoto(with: settings, delegate: self)
    }
    
    // CallBack from take picture
    func capture(_ captureOutput: AVCapturePhotoOutput, didFinishProcessingPhotoSampleBuffer photoSampleBuffer: CMSampleBuffer?, previewPhotoSampleBuffer: CMSampleBuffer?, resolvedSettings: AVCaptureResolvedPhotoSettings, bracketSettings: AVCaptureBracketedStillImageSettings?, error: Error?) {
        
        if let error = error {
            print("error occured : \(error.localizedDescription)")
        }
        
        if  let sampleBuffer = photoSampleBuffer,
            let previewBuffer = previewPhotoSampleBuffer,
            let dataImage =  AVCapturePhotoOutput.jpegPhotoDataRepresentation(forJPEGSampleBuffer:  sampleBuffer, previewPhotoSampleBuffer: previewBuffer) {
            let dataProvider = CGDataProvider(data: dataImage as CFData)
            let cgImageRef: CGImage! = CGImage(jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
            // TODO: Remove setting the UIView image
            self.capturedImage.image = cropToPreviewLayer(originalImage: cgImageRef,
                                                          layer: self.previewView.videoPreviewLayer,
                                                          faceBounds: self.faceBounds)
            faceImage = cropToPreviewLayer(originalImage: cgImageRef,
                                           layer: self.previewView.videoPreviewLayer,
                                           faceBounds: self.faceBounds)
            print(predictEmotion(faceImage: faceImage))
        } else {
            print("some error here")
        }
    }
    
    
    //MARK: - Add image to Library
    var imagePicker: UIImagePickerController!
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            // we got back an error!
            let ac = UIAlertController(title: "Save error", message: error.localizedDescription, preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        } else {
            let ac = UIAlertController(title: "Saved!", message: "Your altered image has been saved to your photos.", preferredStyle: .alert)
            ac.addAction(UIAlertAction(title: "OK", style: .default))
            present(ac, animated: true)
        }
    }
    
    //MARK: - Done image capture here
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        imagePicker.dismiss(animated: true, completion: nil)
        //        imageTake.image = info[UIImagePickerControllerOriginalImage] as? UIImage
    }
    
    // MARK: AVCaptureMetadataOutputObjectsDelegate
    
    func captureOutput(_ captureOutput: AVCaptureOutput!, didOutputMetadataObjects metadataObjects: [Any]!, from connection: AVCaptureConnection!) {
        // wait() is used to drop new notifications if old ones are still processing, to avoid queueing up a bunch of stale data.
        if metadataObjectsOverlayLayersDrawingSemaphore.wait(timeout: DispatchTime.now()) == .success {
            DispatchQueue.main.async { [unowned self] in
                self.removeMetadataObjectOverlayLayers()
                
                var metadataObjectOverlayLayers = [MetadataObjectLayer]()
                for metadataObject in metadataObjects as! [AVMetadataObject] {
                    if metadataObject.type == AVMetadataObjectTypeFace {
                        let metadataObjectOverlayLayer = self.createMetadataObjectOverlayWithMetadataObject(metadataObject)
                        metadataObjectOverlayLayers.append(metadataObjectOverlayLayer)
                    }
                }
                self.addMetadataObjectOverlayLayersToVideoPreviewView(metadataObjectOverlayLayers)
                
                self.metadataObjectsOverlayLayersDrawingSemaphore.signal()
                
                self.captureFace()
            }
        }
    }
}

extension AVCaptureDeviceDiscoverySession
{
    func uniqueDevicePositionsCount() -> Int {
        var uniqueDevicePositions = [AVCaptureDevicePosition]()
        
        for device in devices {
            if !uniqueDevicePositions.contains(device.position) {
                uniqueDevicePositions.append(device.position)
            }
        }
        
        return uniqueDevicePositions.count
    }
}

extension UIDeviceOrientation {
    var videoOrientation: AVCaptureVideoOrientation? {
        switch self {
        case .portrait: return .portrait
        case .portraitUpsideDown: return .portraitUpsideDown
        case .landscapeLeft: return .landscapeRight
        case .landscapeRight: return .landscapeLeft
        default: return nil
        }
    }
}

extension UIInterfaceOrientation {
    var videoOrientation: AVCaptureVideoOrientation? {
        switch self {
        case .portrait: return .portrait
        case .portraitUpsideDown: return .portraitUpsideDown
        case .landscapeLeft: return .landscapeLeft
        case .landscapeRight: return .landscapeRight
        default: return nil
        }
    }
}
