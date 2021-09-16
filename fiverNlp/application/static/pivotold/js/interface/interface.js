let Interface = function (app) {
    this.app = app

    this.setElements()
}

Interface.prototype.setElements = function () {
    let _this = this
    let groupBy = this.app.groupBy
    let keys = this.app.keys
    let extras = this.app.extras
    let elementsExtras = document.getElementById("extras")

    keys.forEach(function (key) {
        let extra = document.createElement("div")

        extra.id = `extra-${key}`
        extra.className = "draggable extra"
        extra.innerHTML = key
        extra.draggable = "true"
        extra.setAttribute("value", key)
        extra.addEventListener("dragstart", function (event) {
            _this.onDragStart(event)
        })

        extra.addEventListener("click", function (event) {
            if (_this.app.extras.includes(key)) {
                document.getElementById(`extra-${key}`).classList.remove("active");
            } else if (!_this.app.extras.includes(key)) {
                document.getElementById(`extra-${key}`).classList.add("active");
            }

            _this.app.setExtras(key)
        })

        elementsExtras.appendChild(extra)

    });

    let elementGroupBy = document.getElementById("group-by")

    elementGroupBy.addEventListener("dragover", function (event) {
        _this.onDragOver(event)
    })
    elementGroupBy.addEventListener("drop", function (event) {
        _this.onDrop(event)
    })

    let elementGroupDump = [...document.getElementsByClassName("group-dump")][0]
    elementGroupDump.addEventListener("dragover", function (event) {
        _this.onDragOverDump(event)
    })

    elementGroupDump.addEventListener("drop", function (event) {
        _this.onDropDump(event)
    })

    groupBy.forEach(function (group) {
        const defaultGroup = document
            .getElementById("extra-" + group)
            .cloneNode(true);

        defaultGroup.addEventListener("dragstart", function (event) {
            _this.onDragStart(event)
        })

        defaultGroup.classList.add("as-group");
        const dropZone = document.getElementById("group-by");
        dropZone.appendChild(defaultGroup);
    })
}

Interface.prototype.onDragStart = function (event) {
    event.dataTransfer.setData("text/plain", event.target.id);
    if (event.target.parentNode.id === "group-by") {
        $(".group-dump").css("display", "block");
    }
}

Interface.prototype.onDragOver = function (event) {
    event.preventDefault();
}

Interface.prototype.onDrop = function (event) {
    let _this = this
    const id = event.dataTransfer.getData("text");
    const draggableElement = document.getElementById(id).cloneNode(true);
    const dropzone = event.currentTarget;

    draggableElement.addEventListener("dragstart", function (event) {
        _this.onDragStart(event)
    })

    const currentGroups = [...document.getElementsByClassName("as-group")].map(element => {
        return element.getAttribute("value")
    })

    const dragElValue = draggableElement.getAttribute("value");

    if (
        draggableElement.classList.contains("as-group") &&
        dropzone.children.length > 1
    ) {
        dropzone.removeChild(document.getElementById(id));
        this.app.removeGroupBy(dragElValue);
        dropzone.appendChild(draggableElement);
        this.app.addGroupBy(dragElValue);
    }

    if (!currentGroups.includes(dragElValue)) {
        draggableElement.classList.add("as-group");
        dropzone.appendChild(draggableElement);
        this.app.addGroupBy(dragElValue);
        event.dataTransfer.clearData();
    }

    $(".group-dump").css("display", "none");
}

Interface.prototype.onDragOverDump = function (event) {
    event.preventDefault();
}

Interface.prototype.updateInterfaceColor = function (colors) {
    let inputGroups = document.getElementsByClassName("as-group")
    for (let item of inputGroups) {
        item.style.backgroundColor = colors(item.getAttribute("value"))
    }
}

Interface.prototype.onDropDump = function (event) {
    const id = event.dataTransfer.getData("text");
    const draggableElement = document.getElementById(id); //.cloneNode(true);
    const parentDragable = document.getElementById("group-by");

    if (
        draggableElement.classList.contains("as-group") &&
        parentDragable.children.length > 1
    ) {
        parentDragable.removeChild(draggableElement);
        this.app.removeGroupBy(draggableElement.getAttribute("value"));
    }

    $(".group-dump").css("display", "none");
}