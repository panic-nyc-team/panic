const BarChart = function (app) {
    this.layout = {}
    this.app = app

    this.setLayout()
    this.draw()
}

BarChart.prototype.draw = function () {
    let _this = this

    this.layerBarChart = this.app.svgBar.append("g").attr("id", "bar-chart");

    // // this.transform()
    this.addBg()

    this.createScales()
    this.setBarWidth()

    this.addXAxis()
    this.addY1Axis()

    this.addBars()

    this.addDataFilter()
    this.addBrushTip()

    this.addChartTitle()
}

BarChart.prototype.createScales = function () {
    this.x = d3
        .scaleLinear()
        .domain([d3.min(this.app.dataRolled.keys()), d3.max(this.app.dataRolled.keys())])
        .rangeRound([0, this.layout.width - this.layout.margin.right - this.layout.margin.left]);

    this.y1 = d3
        .scaleLinear()
        .domain([0, d3.max([...this.app.dataRolled.values()].map(d => d.bundleSize))])
        .range([this.layout.height - this.layout.margin.top - this.layout.margin.bottom, 0]);
}

BarChart.prototype.addXAxis = function () {
    this.xAxis = d3.axisBottom(this.x).ticks(6)//.tickFormat(d3.timeFormat("%b/%d"));

    this.layerBarChart
        .append("g")
        .attr("id", "xAxis")
        .call(this.xAxis)
        .attr(
            "transform",
            `translate(${this.layout.margin.left},${this.layout.height - this.layout.margin.bottom})`)
        .style("color", this.layout.textColor);
}


BarChart.prototype.addY1Axis = function () {
    this.y1Axis = d3.axisLeft(this.y1);
    this.layerBarChart
        .append("g")
        .attr("id", "y1Axis")
        .call(this.y1Axis)
        .attr(
            "transform",
            `translate(${this.layout.margin.left - this.barWidth / 2},${this.layout.margin.top
            })`
        )
        .style("color", this.layout.textColor)
        .call((g) => g.select(".domain").remove());
}

BarChart.prototype.setBarWidth = function () {
    const barNum = d3.max(this.app.dataRolled.keys()) - d3.min(this.app.dataRolled.keys());
    this.barWidth =
        (this.layout.width - this.layout.margin.left - this.layout.margin.right) / barNum;
    this.barPadding = 0.2 * this.barWidth
    this.barWidth = this.barWidth - this.barPadding
}

BarChart.prototype.addBars = function () {
    let _this = this
    this.layerBarChart
        .append("g")
        .attr("fill", "steelblue")
        .attr("fill-opacity", 0.8)
        .selectAll("rect")
        .data([...this.app.dataRolled])
        .join("rect")
        .attr("id", "bars")
        .attr("fill", this.layout.barFill)
        .attr("x", function (d) {
            return _this.layout.margin.left + _this.x(d[0])
        })
        .attr("y", function (d) {
            return _this.layout.margin.top + _this.y1(d[1].bundleSize)
        })
        .attr("width", this.barWidth)
        .attr("height", function (d) {
            return _this.y1(0) - _this.y1(d[1].bundleSize);
        })
        .attr("transform", `translate(${-this.barWidth / 2},0)`)
}

BarChart.prototype.addBg = function () {
    this.layerBarChart
        .append("rect")
        .attr("id", "barchart-bg")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", this.layout.width)
        .attr("height", this.layout.height)
        .attr("fill", this.layout.bgColor);
}

BarChart.prototype.addChartTitle = function () {
    this.layerBarChart
        .append("g")
        .attr("id", "barchart-title")
        .append("foreignObject")
        .attr("width", this.layout.width - this.layout.margin.left - this.layout.margin.right)
        .attr("height", this.layout.margin.top)
        .attr("x", this.layout.margin.left)
        .attr("y", 0)
        .style("font-size", 16)
        .append("xhtml:div")
        .attr("class", "bar-chart-title")
        .append("span")
        .attr("class", "bar-chart-title-span")
        .style("color", this.layout.textColor)
        .html("Similarity Confidence Distribution");
}

BarChart.prototype.addDataFilter = function () {
    let _this = this
    let brushSize = {
        x1: 0,
        x2: this.layout.width - this.layout.margin.right - this.layout.margin.left,
        y1: this.layout.height - this.layout.margin.bottom,
        y2: this.layout.height - this.layout.margin.bottom + 30
    }

    let brush = d3.brushX()
        .extent([[brushSize.x1, brushSize.y1], [brushSize.x2, brushSize.y2]])
        .on("end", brushended);

    this.layerBarChart.append("g").call(brush)
        .call(brush.move, [brushSize.x1, brushSize.x2]).attr(
            "transform",
            `translate(${this.layout.margin.left},${0})`
        )

    function brushended(event) {
        const selection = event.selection;

        if (!event.sourceEvent || !selection) return;

        const [x0, x1] = selection.map(d => Math.round(_this.x.invert(d)));

        d3.select(this)
            .transition()
            .call(brush.move, x1 > x0 ? [x0, x1].map(_this.x) : null);

        _this.app.dataRange.start = x0
        _this.app.dataRange.end = x1

        _this.updateBrushTip()
        _this.app.update()
    }
}

BarChart.prototype.addBrushTip = function () {
    let _this = this

    this.brushTip = this.layerBarChart.append("g").attr("class", "brush-tip").attr(
        "transform",
        `translate(${this.layout.margin.left},${0})`
    )

    let domainX = this.x.domain()

    this.brushTipTexts = this.brushTip
        .selectAll("text")
        .data(domainX)
        .join("text")
        .text(d => d)
        .attr("class", "brush-tips")
        .style("font-size", 10)
        .attr("x", function (d) {
            let textLength = this.getComputedTextLength()
            return _this.x(d) - textLength / 2
        })
        .attr("y", this.layout.height + 20)
        .attr("fill", this.layout.textColor)

}

BarChart.prototype.updateBrushTip = function () {
    let _this = this
    let newTips = [this.app.dataRange.start, this.app.dataRange.end]

    let format = d3.timeFormat("%b/%d")

    this.brushTipTexts
        .data(newTips)
        .join("text")
        .text(d => d)
        .attr("x", function (d) {
            let textLength = this.getComputedTextLength()
            return _this.x(d) - textLength / 2
        })

        .attr("x", function (d) {
            let textLength = this.getComputedTextLength()
            return _this.x(d) - textLength / 2
        })

}







BarChart.prototype.transform = function () {
    this.app.svg2.attr(
        "transform",
        `translate(${window.innerWidth * 0.55},${0})`
    );
}

BarChart.prototype.updateDarkMode = function () {
    d3.selectAll("#bars").attr("fill", this.layout.barFill)
    d3.select("#xAxis").style("color", this.layout.textColor)
    d3.select("#y1Axis").style("color", this.layout.textColor)
    d3.select("#y2Axis").style("color", this.layout.textColor)
    d3.select("#barchart-bg").attr("fill", this.layout.bgColor)
    d3.select(".bar-chart-title-span").style("color", this.layout.textColor)
    d3.selectAll(".brush-tips").attr("fill", this.layout.textColor)
}

BarChart.prototype.setLayout = function () {
    let _this = this
    this.layout = {
        height: 200,
        width: 500,
        margin: {
            top: 50,
            bottom: 30,
            left: 50,
            right: 50,
        },
        barFill: function () { return _this.app.darkMode ? "#aaa" : "#333" },
        lineStroke: function () { return _this.app.darkMode ? "#3978e6" : "#3978e6" },
        bgColor: function () { return _this.app.darkMode ? "#222" : "#fff" },
        textColor: function () { return _this.app.darkMode ? "#fff" : "#111" },
    };
}
