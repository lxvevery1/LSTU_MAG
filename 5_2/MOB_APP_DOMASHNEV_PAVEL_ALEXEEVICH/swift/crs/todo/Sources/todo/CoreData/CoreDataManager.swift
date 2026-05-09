import CoreData

final class CoreDataManager {

    static let shared = CoreDataManager()

    lazy var container: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "TodoModel")
        container.loadPersistentStores { _, error in
            if let error { fatalError("\(error)") }
        }
        return container
    }()

    var context: NSManagedObjectContext {
        container.viewContext
    }

    func save() {
        try? context.save()
    }
}
