//
//  SongManager.swift
//  EmotionHero
//
//  Created by Chris Kalas on 24.06.17.
//  Copyright © 2017 Chris Kalas. All rights reserved.
//

import Foundation
import SpriteKit

// TODO: have speed as a separate setting

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
        speed = 2
        
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
        currentSong = song
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
