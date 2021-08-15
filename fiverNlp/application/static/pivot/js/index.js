let App = function (rawData) {
    this.rawData = rawData
    this.data = null
    this.keys = Object.keys(this.rawData[0]).filter((d) => d !== "type");

    this.dataRange = { start: null, end: null }

    this.groupBy = ["site_type", "country"]; // Set default hiearchy attribute
    this.extras = [];

    this.darkMode = true;

    this.prepareData()
    this.setData()

    this.addSvg()

    this.interface = new Interface(this)
    this.pivotChart = new PivotChart(this)
    this.barChart = new BarChart(this)

    this.interface.updateInterfaceColor(this.pivotChart.treeGraph.treeColors)
    this.handleDarkMode()
}

App.prototype.addSvg = function () {
    this.svg = d3
        .select("#chart")
        .append("svg")
        .attr("class", ".pivot-chart-svg")
        .style("position", "absolute")
        .style("z-index", "-1")
        .style("top", 0)
        .style("left", 0)
        .style("width", "100%")
        .style("height", "100%")
        .style("background-color", "white");

    this.svg2 = d3
        .select("#chart2")
        .append("svg")
        .attr("class", ".bar-chart-svg")
        .style("position", "absolute")
        .style("z-index", "-1")
        .style("top", 0)
        .style("left", 0)
        .style("width", 500)
        .style("height", 230)
        .style("background-color", "none");
}

App.prototype.addGroupBy = function (value) {
    this.groupBy.push(value)
    this.pivotChart.updateChart()
    this.interface.updateInterfaceColor(this.pivotChart.treeGraph.treeColors)
}

App.prototype.removeGroupBy = function (value) {
    this.groupBy = this.groupBy.filter((d) => d !== value);
    this.pivotChart.updateChart()

    this.interface.updateInterfaceColor(this.pivotChart.treeGraph.treeColors)
};

App.prototype.setExtras = function (k) {
    if (this.extras.includes(k)) {
        this.extras = this.extras.filter((v) => v !== k);
    } else {
        this.extras.push(k);
    }
    this.pivotChart.updateChartSide()
};

App.prototype.setData = function () {
    this.data = this.rawData
}

App.prototype.prepareData = function () {
    let _this = this

    this.rawData.forEach(function (node, i) {
        _this.keys.forEach((k) => {
            if (node[k] === null) {
                _this.rawData[i][k] = "null";
            } else if (node[k] === undefined) {
                _this.rawData[i][k] = "null";
            }
            _this.rawData[i]["type"] = "main";
        });

        const date = new Date(node.publish_date);
        _this.rawData[i].date_published = date;
        _this.rawData[i].date_string = date.toDateString();
    });

    this.dataRolled = d3.rollup(
        this.rawData,
        (v) => {
            const random_num = 1 + Math.random() * 2;
            return {
                length: v.length,
                closing_price: random_num.toFixed(2),
                bundle: v,
            };
        },
        (d) => new Date(d.date_string)
    );
}

App.prototype.updateData = function () {

    this.data = this.filterByDate(this.rawData, this.dataRange)

    this.pivotChart.restartChart()
}

App.prototype.filterByDate = function (data, range) {
    let start = range.start.getTime()
    let end = range.end.getTime()

    return data.filter(function (d) {
        let nodeDate = d.date_published.getTime()
        return start < nodeDate && nodeDate < end
    })
}



App.prototype.handleDarkMode = function () {
    let _this = this
    const toggleDark = d3.select("#toggle-dark");
    const localStorage = window.localStorage;

    if (localStorage.pivotChartDarkMode === undefined) {
        console.log(this)
        this.darkMode = this.darkMode
    } else {
        this.darkMode = this.darkMode === "true" ? true : false;
    } toggleDark.node().checked = this.darkMode;
    this.setDarkMode();

    toggleDark.on("click", function (e) {
        _this.darkMode = this.checked;
        window.localStorage.pivotChartDarkMode = _this.darkMode;
        _this.setDarkMode();
    });
}

App.prototype.setDarkMode = function () {
    this.pivotChart.updateDarkMode()
    this.barChart.updateDarkMode()
}


d3.json(query_name)
    .then(function (json) {
        var app = new App(json)

    })
    .catch(function (error) {
        console.log(error);
    });