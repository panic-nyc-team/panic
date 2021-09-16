let DocumentList = function (app) {
    this.app = app
    this.displayState = "none"
    this.setUp()
}

DocumentList.prototype.setUp = function () {
    let _this = this
    this.list = d3.select("#document-list").style("display", this.displayState)
    this.ul = this.list.append("ul").attr("class", "document-elements")

    this.setLayout()
    this.setPosition()
}

DocumentList.prototype.setPosition = function (params) {
    let interface = d3.select("#interface").node().getBoundingClientRect()

    this.list
        .style("top", `${interface.bottom + 20}px`)
        .style("left", "50px")
    this.ul
        .style("max-height", `${window.innerHeight / 2 - 150}px`)
}

DocumentList.prototype.render = function ({ data, displayState }) {
    const _this = this
    const searchInput = d3.select("#document-search-input")

    this.list.style("display", displayState)

    this.ul.selectAll("li")
        .data(data, d => d)
        .join(
            (enter) => enter.append("li")
                .attr("id", (d, i) => `document-li-${i}`)
                .attr('class', "document-li")
                .style("background-color", function () {
                    return _this.layout.liBgColor()
                })
                .html(d => elementHtml(d))
            ,
            update => update
        )

    searchInput
        .style("background-color", this.layout.searchColor)
        .on("keyup", function (e) {
            const rgx = `${searchInput.node().value}`
            const pattern = new RegExp(rgx, "i")
            const list = d3.selectAll(".document-li")
            list.nodes().forEach(element => {
                if (pattern.test(element.innerText)) {
                    d3.select(`#${element.id}`).style("display", "block")
                } else {
                    d3.select(`#${element.id}`).style("display", "none")
                }
            });
        })

    function elementHtml({ title, url }) {
        return `
               <a href=${url} target="_blank"><p>${title}</p></a>
            `
    }
}

DocumentList.prototype.updateDarkMode = function () {
    d3.selectAll(".document-li").style("background-color", this.layout.liBgColor)
    d3.select("#document-search-input").style("background-color", this.layout.searchColor)
}

DocumentList.prototype.setLayout = function () {
    let _this = this
    this.layout = {
        liBgColor: function () { return _this.app.darkMode ? "#fff" : "#eee" },
        searchColor: function () { return _this.app.darkMode ? "#fff" : "#ddd" },
    }
}


DocumentList.prototype.elementLayout = function (title) {

}
