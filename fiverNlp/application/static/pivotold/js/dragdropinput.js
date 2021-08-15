function onDragStart(event) {
  event.dataTransfer.setData("text/plain", event.target.id);
  if (event.target.parentNode.id === "group-by") {
    $(".group-dump").css("display", "block");
  }
  // const node = event.target;
  // const parentNode = event.target.parentNode;
  // if (parentNode.id === "group-by" && parentNode.childNodes.length > 1) {
  //   parentNode.removeChild(node);
  //   this.removeGroupBy(node.getAttribute("value"));
  // }
}

function onDragOver(event) {
  event.preventDefault();
}

function onDrop(event) {
  const id = event.dataTransfer.getData("text");
  const draggableElement = document.getElementById(id).cloneNode(true);
  const dropzone = event.currentTarget;

  const currentGroups = d3
    .selectAll(".as-group")
    .nodes()
    .map((d) => d.attributes.value.value);

  const dragElValue = draggableElement.getAttribute("value");

  if (
    draggableElement.classList.contains("as-group") &&
    dropzone.children.length > 1
  ) {
    dropzone.removeChild(document.getElementById(id));
    this.removeGroupBy(dragElValue);
    dropzone.appendChild(draggableElement);
    this.addGroupBy(dragElValue);
  }

  if (!currentGroups.includes(dragElValue)) {
    draggableElement.classList.add("as-group");
    dropzone.appendChild(draggableElement);
    this.addGroupBy(dragElValue);
    event.dataTransfer.clearData();
  }

  $(".group-dump").css("display", "none");
}

function onDragOverDump(event) {
  event.preventDefault();
}

function onDropDump(event) {
  const id = event.dataTransfer.getData("text");
  const draggableElement = document.getElementById(id); //.cloneNode(true);
  const parentDragable = document.getElementById("group-by");

  if (
    draggableElement.classList.contains("as-group") &&
    parentDragable.children.length > 1
  ) {
    parentDragable.removeChild(draggableElement);
    this.removeGroupBy(draggableElement.getAttribute("value"));
  }

  $(".group-dump").css("display", "none");
}

function manageInputs(groupBy) {
  d3.select("#extras")
    .selectAll("div")
    .data(this.keys)
    .join("div")
    .attr("id", (k) => k)
    .text((k) => k)
    .attr("class", "draggable extra")
    .attr("id", (k) => "extra-" + k)
    .attr("value", (k) => k)
    .attr("draggable", "true")
    .attr("ondragstart", "onDragStart(event)")
    .on("click", (event, k) => {
      if (this.extras.includes(k)) {
        d3.select(`#extra-${k}`).classed("active", false);
      } else {
        d3.select(`#extra-${k}`).classed("active", true);
      }
      this.setExtras(k);
    });

  d3.select("#group-by")
    .attr("ondragover", "onDragOver(event)")
    .attr("ondrop", "onDrop(event)");

  //Set GroupBy default
  //
  d3.select(".group-dump")
    .attr("ondragover", "onDragOverDump(event)")
    .attr("ondrop", "onDropDump(event)");

  for (let group of groupBy) {
    const defaultGroup = document
      .getElementById("extra-" + group)
      .cloneNode(true);
    defaultGroup.classList.add("as-group");
    const dropZone = document.getElementById("group-by");
    dropZone.appendChild(defaultGroup);
  }
}
