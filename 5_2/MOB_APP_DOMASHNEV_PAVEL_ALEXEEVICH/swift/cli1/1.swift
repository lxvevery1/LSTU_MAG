//
//  main.swift
//  Task1
//
//  Created by MAC ASU on 06/11/2020.
//  Copyright © 2020 LSTU. All rights reserved.
//

print("Введите размер массива")


// selection sort
func sort(arr: inout [(article: Int, price: Float, name: String)],
          comparer: ((article: Int, price: Float, name: String),
          (article: Int, price: Float, name: String)) -> Int )
{
    let n = arr.count

    for i in 0..<n {
        var minId = i
        for j in i+1..<n {
            if comparer(arr[j], arr[minId]) > 0 {
                minId = j
            }
        }

        if minId != i {
            arr.swapAt(i, minId)
        }
    }
}

func printArr(arr: [(article: Int, price: Float, name: String)]) {
    for elem in arr {
        print(elem)
    }
}


var arr = [(article: Int, price: Float, name: String)]()

let count = Int(readLine() ?? "0") ?? 0

for _ in 0..<count {
    let article = Int.random(in: 1...100)
    let price = Float.random(in: 10.0...1000.0)
    let name = "Product \(article)"

    arr.append((article, price, name ))
}

print("Исходный массив")
printArr(arr: arr)

var selection = 0

while selection == 0 {
    print("Выберите по какому полю сортировать (1 - по артикулу, 2 - по цене)")
    selection = Int(readLine() ?? "") ?? 0
}

switch selection {
case 1: print("Сортируем по артикулу")
        sort(arr: &arr)
        { left, right in
            if left.article == right.article {
                return 0
            }
            else if left.article > right.article {
                return -1
            }
            else {
                return 1
            }
        }
case 2: print("Сортируем по цене")
        sort(arr: &arr)
        { left, right in
            if left.price == right.price {
                return 0
            }
            else if left.price > right.price {
                return -1
            }
            else {
                return 1
            }
        }
default: print("не сортируем")
}

print("Упорядоченный массив")
printArr(arr: arr)


// find a price item in sorted array
func find_price(arr: inout [(article: Int, price: Float, name: String)],
          target: Float )  -> Int?
{
    var left = 0
    var right = arr.count-1

    while left <= right {
        let mid = (left + right) / 2
        let midValue = arr[mid].price

        if midValue == target {
            return mid
        }
        else if midValue < target {
            left = mid + 1
        }
        else {
            right = mid - 1
        }
    }

    return nil
}

// find an article item in sorted array
func find_article(arr: inout [(article: Int, price: Float, name: String)],
          target: Int) -> Int?
{
    var left = 0
    var right = arr.count-1

    while left <= right {
        let mid = (left + right) / 2
        let midValue = arr[mid].article

        if midValue == target {
            return mid
        }
        else if midValue < target {
            left = mid + 1
        }
        else {
            right = mid - 1
        }
    }

    return nil
}

// now find an element in sorted array
print("Введите искомое значение")
let search_input = readLine() ?? ""

switch selection {

case 1: print("Ищем артикул...")

        if let article_val = Int(search_input),
            let index = find_article(arr: &arr, target: article_val) {
                print("Найден артикул на позиции \(index)")
                print(arr[index])
        }
        else {
            print("Артикул не найден")
        }

case 2: print("Ищем цену...")

        if let price_val = Float(search_input),
            let index = find_price(arr: &arr, target: price_val) {
                print("Найдена цена на позиции \(index)")
                print(arr[index])
        }
        else {
            print("Цена не найдена")
        }
default: print("не ищем")
}
