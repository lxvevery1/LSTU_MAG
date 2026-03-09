/* Задание: написать консольную игру-ходилку с кубиком и фишками для 2-4
 * игроков; параметры игры и типы используемых секретных клеток приведены в
 * таблице №1. */

/* Программа должна выполнять следующее: 1. Генерация поля игры заданного
 * размера (использовать паттерн «data source») 2. Выбор количества игроков 3.
 * Текстовое отображение процесса игры (использовать паттерн «delegate») */

/* Секретные клетки: 1. Кол-во кубиков = 1, 2. Типы секретных клеток:
 * дополнительный, пропуск хода. */

import Foundation

let PLAYER_COUNT_MIN: Int = 2
let PLAYER_COUNT_MAX: Int = 4

let GAMEMAP_SIZE: Int = 10
let SECRET_CELL_CHANCE: Float = 0.1

protocol GameMapDataSource {
    func generate_map(size: Int) -> [Cell]
}

protocol PlayerDataSource {
    func init_players() -> [Player]
}

protocol GameDelegate {
    func render(game: Game)
}

class ConsoleRenderer: GameDelegate {
    func render(game: Game) {
        for i in 0..<game.map.size {

            var printed = false

            for p in game.players {
                if p.position == i {
                    print("P\(p.id)", terminator: " ")
                    printed = true
                }
            }

            if !printed {
                print(game.map.cells[i].render, terminator: " ")
            }
        }

        print("\n")
    }
}

class GameMap: GameMapDataSource {

    var size: Int
    var cells: [Cell] = []

    init(size: Int) {
        self.size = size
    }

    func generate_map(size: Int) -> [Cell] {

        var cells = [Cell]()

        for _ in 0..<size {
            cells.append(Cell(type: .normal))
        }

        for _ in 0..<Int(Float(size) * SECRET_CELL_CHANCE) {
            let index = Int.random(in: 1..<size-1)
            cells[index] = Cell(type: .extra_dice)
        }

        for _ in 0..<Int(Float(size) * SECRET_CELL_CHANCE) {
            let index = Int.random(in: 1..<size-1)
            cells[index] = Cell(type: .skip_turn)
        }

        return cells
    }
}

class Game: PlayerDataSource {

    var turn = 0
    var turnOfPlayer = 0

    var players = [Player]()
    var map: GameMap
    var winner: Player?

    var delegate: GameDelegate?

    init() {

        map = GameMap(size: GAMEMAP_SIZE)

        players = init_players()

        map.cells = map.generate_map(size: GAMEMAP_SIZE)

        delegate = ConsoleRenderer()

        print("Map size: \(map.size)")

        delegate?.render(game: self)

        loop()
    }

    func init_players() -> [Player] {

        let count = player_count_selection()
        var result = [Player]()

        for i in 0..<count {
            result.append(Player(position: 0, id: i, name: "Player"))
        }

        return result
    }

    func player_count_selection() -> Int {

        print("Enter player count (\(PLAYER_COUNT_MIN)-\(PLAYER_COUNT_MAX))")

        if let input = readLine(), let count = Int(input) {

            if count >= PLAYER_COUNT_MIN && count <= PLAYER_COUNT_MAX {
                return count
            }
        }

        print("Invalid input")
        return player_count_selection()
    }

    func loop() {

        while winner == nil {

            next_turn()

            for player in players {
                if player.position >= map.size - 1 {
                    winner = player
                    break
                }
            }

            delegate?.render(game: self)
        }

        print("Winner: \(winner!.get_name())")
    }

    func next_turn() {

        let player = players[turnOfPlayer]

        if player.skipNextTurn {
            print("\(player.get_name()) skips turn")
            player.skipNextTurn = false
        } else {

            player.step()

            if player.position >= map.size {
                player.position = map.size - 1
            }

            let cell = map.cells[player.position]

            switch cell.type {

            case .extra_dice:
                print("Extra dice!")
                player.step()

            case .skip_turn:
                print("Next turn skipped!")
                player.skipNextTurn = true

            default:
                break
            }
        }

        turn += 1
        turnOfPlayer = (turnOfPlayer + 1) % players.count

        print("Turn \(turn)\n")
    }
}

class Cell {

    var type: CellType
    var render: Character

    init(type: CellType) {

        self.type = type

        switch type {

        case .normal:
            render = "."

        case .extra_dice:
            render = "E"

        case .skip_turn:
            render = "S"

        }
    }
}

class Player {

    let DICE_VALUE_MIN = 1
    let DICE_VALUE_MAX = 6

    var position: Int
    var id: Int
    var name: String

    var skipNextTurn = false

    init(position: Int, id: Int, name: String) {
        self.position = position
        self.id = id
        self.name = name
    }

    func step() {

        let dice = roll_dice()
        let last = position

        position += dice

        print("\(get_name()) rolled \(dice)")
        print("\(get_name()) moved from \(last) to \(position)")
    }

    func roll_dice() -> Int {
        return Int.random(in: DICE_VALUE_MIN...DICE_VALUE_MAX)
    }

    func get_name() -> String {
        return name + "\(id)"
    }
}

enum CellType {
    case normal
    case extra_dice
    case skip_turn
}

var game = Game()
