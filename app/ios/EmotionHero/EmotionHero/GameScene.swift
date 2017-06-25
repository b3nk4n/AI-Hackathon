//
//  GameScene.swift
//  EmotionHero
//
//  Created by Chris Kalas on 24.06.17.
//  Copyright © 2017 Chris Kalas. All rights reserved.
//

import Foundation
import SpriteKit


class GameScene: SKScene {
    
    let emotionFiles = ["emoji_faces/1f60a.png","emoji_faces/1f61f.png","emoji_faces/1f62e.png","emoji_faces/1f61c.png","emoji_faces/1f61b.png"]
    let sm = SongManager()
    var grid: Grid?
    var gridLength: Int
    var timeStep: Int
    
    var spriteList: [(sprite: SKSpriteNode, position: Int, lane: Int)]
    
    override init(size: CGSize) {
        timeStep = 0
        gridLength = 5
        spriteList = [(sprite: SKSpriteNode, position: Int, lane: Int)]()
        
        super.init(size: size)
        
        grid = Grid(blockSize: 70.0, rows: gridLength, cols: 4)
        
        grid!.position = CGPoint (x:frame.midX, y:frame.midY)
        addChild(grid!)
        /*for point: CGPoint in [CGPoint(x:-104.5, y:-139.5),CGPoint(x:-34.5, y:-139.5),CGPoint(x:35.5, y:-139.5),CGPoint(x:105.5, y:-139.5)] {
            let tmp = SKSpriteNode(imageNamed: "emoji_faces/1f643.png")
            tmp.alpha = 0.4
            tmp.setScale(0.45)
            tmp.position = point
            addChild(tmp)
        }*/
        // TODO: better way of managing song
        sm.generateRandomSong(length: 200, difficulty: .Medium)
        
        
    }
    
    func playSong(song: Song) {
        sm.currentSong = song
        
        // Put progress bar whilst loading
        for type in sm.currentSong.sequence {
            let tmp = SKSpriteNode(imageNamed: emotionFiles[type])
            let lane = Int(arc4random_uniform(UInt32(4)))
            //tmp.addGlow()
            tmp.zPosition = 100
            tmp.position = grid!.gridPosition(row: 0, col: lane)
            tmp.setScale(0.45)
            tmp.isHidden = true
            grid!.addChild(tmp)
        }
        
        // TODO: fix song speed in init
        let wait = SKAction.wait(forDuration: sm.currentSong.speed)
        
        let action = SKAction.run { [unowned self] in
            self.grid!.children[self.timeStep].run(SKAction.sequence([SKAction.moveTo(y: self.grid!.gridPosition(row: self.gridLength, col: 0).y, duration: 3),SKAction.hide()]))
            self.grid!.children[self.timeStep].isHidden = false
            //(self.grid!.children[self.timeStep] as! SKSpriteNode).dismissCorrect()
            self.timeStep += 1
        }
        
        let seq = SKAction.sequence([action, wait])
        
        let repeater = SKAction.repeat(seq, count: sm.currentSong.length)
        run(repeater)
        
    }
    
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func evaluateExpression() -> Bool {
        return true
    }
    
    // center of bottom row is ~-143
    override func update(_ currentTime: TimeInterval) {
        for child in grid!.children {
            if child.isHidden == false {
                //print(child.position.y)
                if child.position.y < -80 && child.position.y > -84{
                    (child as! SKSpriteNode).emphasize()
                }
            }
        }
            
    }
    
    override func didMove(to: SKView) {
        
    }
    
    
}

extension SKSpriteNode {
    
    func addGlow(radius: Float = 30) {
        self.removeAllChildren()
        let effectNode = SKEffectNode()
        effectNode.shouldRasterize = true
        addChild(effectNode)
        effectNode.addChild(SKSpriteNode(texture: texture))
        effectNode.filter = CIFilter(name: "CIGaussianBlur", withInputParameters: ["inputRadius":radius])
    }
    
    func addGreenGlow(radius: Float = 60) {
        self.removeAllChildren()
        let effectNode = SKEffectNode()
        effectNode.shouldRasterize = true
        addChild(effectNode)
        let greenGlow = SKTexture(imageNamed: "emoji_faces/1f922.png")
        effectNode.addChild(SKSpriteNode(texture: greenGlow))
        effectNode.filter = CIFilter(name: "CIGaussianBlur", withInputParameters: ["inputRadius":radius])
    }
    
    func addRedGlow(radius: Float = 60) {
        self.removeAllChildren()
        let effectNode = SKEffectNode()
        effectNode.shouldRasterize = true
        addChild(effectNode)
        let greenGlow = SKTexture(imageNamed: "emoji_faces/1f621.png")
        effectNode.addChild(SKSpriteNode(texture: greenGlow))
        effectNode.filter = CIFilter(name: "CIGaussianBlur", withInputParameters: ["inputRadius":radius])
    }
    
    func dismissCorrect() {
        self.removeAllChildren()
        let dismissTime = 0.5
        let expand = SKAction.scale(to: 1.5, duration: dismissTime)
        let clear = SKAction.fadeAlpha(to: 0, duration: dismissTime)
        self.run(SKAction.group([expand, clear]))
    }
    
    func emphasize(radius: Float = 30) {
        let effectNode = SKEffectNode()
        effectNode.shouldRasterize = true
        addChild(effectNode)
        let glow = SKTexture(imageNamed: "emoji_faces/1f535.png" )
        effectNode.addChild(SKSpriteNode(texture: glow))
        effectNode.filter = CIFilter(name: "CIGaussianBlur", withInputParameters: ["inputRadius":radius])
        
        let expand = SKAction.scale(to: 0.7, duration: 0.1)
        self.run(expand)

    }
}