//: Playground - noun: a place where people can play

import UIKit
import SpriteKit
import PlaygroundSupport
//PlaygroundPage.current.needsIndefiniteExecution = true

struct Song {
    var sequence: [Int]
    // speed: 0-100 ?? the time interval at which to fire the get emotion event
    var speed: TimeInterval
    var difficulty: Difficulty?
    
    var length: Int  {
        return sequence.count
    }
    
    
    // TODO: add some init with difficulty
    init() {
        sequence = [Int]()
        speed = 1
        
    }
    

    init(sequence:[Int], speed: TimeInterval) {
        self.sequence = sequence
        self.speed = speed
    }
    
}

enum Difficulty: Int {
    case Easy = 3
    case Medium = 5
    case Hard = 7
}

class SongManager {
    
    // Dictionary of songs indexed by difficulty
    var songs: [Difficulty: [Song]]
    var currentSong: Song
    
    init() {
        songs = [Difficulty: [Song]]()
        songs[.Easy] = [Song]()
        songs[.Medium] = [Song]()
        songs[.Hard] = [Song]()
        currentSong = Song()
    }
    
    func generateRandomSong(length: Int, difficulty: Difficulty) {
        var song = Song()
        song.difficulty = difficulty
        for _ in 0..<length {
            song.sequence.append(Int(arc4random_uniform(UInt32(difficulty.rawValue))))
        }

        songs[difficulty]!.append(song)
    }
    
    func getSongs() -> [Difficulty: [Song]]{
        return songs
    }
    
    
    func displaySong(song: Song) {
        print("Displaying Song:")
        print("Speed: \(song.speed), Length: \(song.length)")
        if let difficulty = song.difficulty {
            print("Difficulty: \(difficulty)")
        }
        print("Sequence: \(song.sequence)")
    }
    

    // TODO: think about current song management
    
    func setSong(song: Song) {
        currentSong = song
    }

    
}


// GRID SYSTEM

class Grid:SKSpriteNode {
    var rows:Int!
    var cols:Int!
    var blockSize:CGFloat!
    
    convenience init?(blockSize:CGFloat,rows:Int,cols:Int) {
        guard let texture = Grid.gridTexture(blockSize: blockSize,rows: rows, cols:cols) else {
            return nil
        }
        self.init(texture: texture, color:SKColor.clear, size: texture.size())
        self.blockSize = blockSize
        self.rows = rows
        self.cols = cols
    }
    
    class func gridTexture(blockSize:CGFloat,rows:Int,cols:Int) -> SKTexture? {
        // Add 1 to the height and width to ensure the borders are within the sprite
        let size = CGSize(width: CGFloat(cols)*blockSize+1.0, height: CGFloat(rows)*blockSize+1.0)
        UIGraphicsBeginImageContext(size)
        
        guard let context = UIGraphicsGetCurrentContext() else {
            return nil
        }
        
        let bezierPath = UIBezierPath()
        let offset:CGFloat = 0.5
        // Draw vertical lines
        for i in 0...cols {
            let x = CGFloat(i)*blockSize + offset
            bezierPath.move(to: CGPoint(x: x, y: 0))
            bezierPath.addLine(to: CGPoint(x: x, y: size.height))
        }
        // Draw horizontal lines
        for i in 0...rows {
            let y = CGFloat(i)*blockSize + offset
            bezierPath.move(to: CGPoint(x: 0, y: y))
            bezierPath.addLine(to: CGPoint(x: size.width, y: y))
        }
        SKColor.white.setStroke()
        bezierPath.lineWidth = 1.0
        bezierPath.stroke()
        context.addPath(bezierPath.cgPath)
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return SKTexture(image: image!)
    }
    
    func gridPosition(row:Int, col:Int) -> CGPoint {
        let offset = blockSize / 2.0 + 0.5
        let x = CGFloat(col) * blockSize - (blockSize * CGFloat(cols)) / 2.0 + offset
        let y = CGFloat(rows - row - 1) * blockSize - (blockSize * CGFloat(rows)) / 2.0 + offset
        return CGPoint(x:x, y:y)
    }
}


class GameScene: SKScene {
    
    let emotionFiles = ["1f60a.png","1f61f.png","1f62e.png","1f61c.png","1f61b.png"]
    let sm = SongManager()
    var timer: Timer?
    var grid: Grid?
    var gridLength: Int
    var timeStep: Int
    var loadedSprites: [SKSpriteNode]

    var spriteList: [(sprite: SKSpriteNode, position: Int, lane: Int)]

    override init(size: CGSize) {
        timeStep = 0
        gridLength = 5
        spriteList = [(sprite: SKSpriteNode, position: Int, lane: Int)]()
        loadedSprites = [SKSpriteNode]()
        for file in emotionFiles {
            loadedSprites.append(SKSpriteNode(imageNamed: file))
        }
        super.init(size: size)
        
        grid = Grid(blockSize: 70.0, rows: gridLength, cols: 4)
        
        grid!.position = CGPoint (x:frame.midX, y:frame.midY)
        addChild(grid!)
        
        // TODO: better way of managing song
        sm.generateRandomSong(length: 50, difficulty: .Medium)
        
    
    }
    

    func playSong(song: Song) {
        sm.currentSong = song
        
        // Put progress bar whilst loading
        for type in sm.currentSong.sequence {
            let tmp = SKSpriteNode(imageNamed: emotionFiles[type])
            let lane = Int(arc4random_uniform(UInt32(4)))
            tmp.position = grid!.gridPosition(row: 0, col: lane)
            tmp.setScale(0.45)
            tmp.isHidden = true
            grid!.addChild(tmp)
        }
        
        // TODO: fix song speed in init
        let wait = SKAction.wait(forDuration: 1)//sm.currentSong.speed)
        
        let action = SKAction.run { [unowned self] in
            self.grid!.children[self.timeStep].run(SKAction.moveTo(y: self.grid!.gridPosition(row: self.gridLength, col: 0).y, duration: 3))
            self.grid!.children[self.timeStep].isHidden = false
            self.timeStep += 1
        }

        let seq = SKAction.sequence([action, wait])
        
        let repeater = SKAction.repeat(seq, count: UInt(sm.currentSong.length))
        run(repeater)
        // TODO: Fix this
        
    }
    
        
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func update(_ currentTime: TimeInterval) {
        /*var removeList = [Int]()
        let endY = grid!.gridPosition(row: gridLength, col: 0).y
        // Remove old nodes
        for (i,_) in grid!.children.enumerated() {
            if grid!.children[i].position.y == endY {
                removeList.append(i)
            }
        }
        
        for i in removeList {
            grid!.children[i].removeFromParent()
        }
        
        print("\(grid!.children.count) nodes on graph")*/
        
    }
    
    override func didMove(to: SKView) {

    }
    
    
}


//Create the SpriteKit View
let sceneView = SKView(frame: CGRect(x:0 , y:0, width: 500, height: 500))

let scene = GameScene(size: sceneView.frame.size)
scene.playSong(song: scene.sm.songs[.Medium]![0])
sceneView.showsFPS = true
sceneView.presentScene(scene)
PlaygroundPage.current.liveView = sceneView

