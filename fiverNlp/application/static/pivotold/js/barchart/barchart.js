let BarChart = function (app) {
  this.layout = {}
  this.app = app

  this.setLayout()
  this.draw()
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
      return _this.layout.margin.left + _this.x(d[0]) - _this.barWidth / 2
    })
    .attr("y", function (d) {
      return _this.layout.margin.top + _this.y1(d[1].length)
    })
    .attr("width", this.barWidth)
    .attr("height", function (d) {
      return _this.y1(0) - _this.y1(d[1].length);
    })
}

BarChart.prototype.addLine = function () {
  let _this = this
  const line = d3
    .line()
    .x(function (d) {
      return _this.x(d[0]) + _this.layout.margin.left;
    })
    .y((d) => {
      return _this.y2(d[1].closing_price) + _this.layout.margin.top;
    });


  this.layerBarChart
    .append("path")
    .attr("fill", "none")
    .attr("stroke", this.layout.lineStroke)
    .attr("stroke-miterlimit", 1)
    .attr("stroke-width", 2)
    .attr("d", line([...this.app.dataRolled].sort((a, b) => a[0] - b[0])));
}



BarChart.prototype.draw = function () {
  let _this = this

  this.layerBarChart = this.app.svg2.append("g").attr("id", "bar-chart");

  this.transform()
  this.addBg()

  this.setCounts()
  this.setClosingPrice()

  this.createScales()
  this.setBarWidth()

  this.addXAxis()
  this.addY1Axis()
  // this.addY2Axis()

  this.addBars()
  // this.addLine()

  this.addDataFilter()
  this.addBrushTip()

  this.addChartTitle()
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
    .html("Daily Document Volume");
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
    const interval = d3.timeHour.every(24)

    if (!event.sourceEvent || !selection) return;

    const [x0, x1] = selection.map(d => interval.round(_this.x.invert(d)));


    d3.select(this)
      .transition()
      .call(brush.move, x1 > x0 ? [x0, x1].map(_this.x) : null);

    _this.app.dataRange.start = x0
    _this.app.dataRange.end = x1

    _this.updateBrushTip()
    _this.app.updateData()
  }
}

BarChart.prototype.addBrushTip = function () {
  let _this = this

  this.brushTip = this.layerBarChart.append("g").attr("class", "brush-tip").attr(
    "transform",
    `translate(${this.layout.margin.left},${0})`
  )

  let format = d3.timeFormat("%b/%d")

  let dates = this.x.domain()

  this.brushTipTexts = this.brushTip
    .selectAll("text")
    .data(dates)
    .join("text")
    .text(function (d) {
      return format(d)
    })
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
  let newDates = [this.app.dataRange.start, this.app.dataRange.end]

  let format = d3.timeFormat("%b/%d")

  this.brushTipTexts
    .data(newDates)
    .join("text")
    .text(function (d) {
      return format(d)
    })
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

BarChart.prototype.setCounts = function () {
  this.counts = [...this.app.dataRolled.values()].map((d) => d.length);
}

BarChart.prototype.setClosingPrice = function () {
  this.closingPrice = [...this.app.dataRolled.values()].map((d) => d.closing_price);
}

BarChart.prototype.createScales = function () {
  this.x = d3
    .scaleTime()
    .domain([d3.min(this.app.dataRolled.keys()), d3.max(this.app.dataRolled.keys())])
    .rangeRound([0, this.layout.width - this.layout.margin.right - this.layout.margin.left]);

  this.y1 = d3
    .scaleLinear()
    .domain([0, d3.max(this.counts)])
    .range([this.layout.height - this.layout.margin.top - this.layout.margin.bottom, 0]);

  this.y2 = d3
    .scaleLinear()
    .domain([d3.min(this.closingPrice), d3.max(this.closingPrice)])
    .range([this.layout.height - this.layout.margin.top - this.layout.margin.bottom, 0]);
}

BarChart.prototype.setBarWidth = function () {
  const oneDay = 24 * 60 * 60 * 1000;
  const dateDiffMs = d3.max(this.app.dataRolled.keys()) - d3.min(this.app.dataRolled.keys());
  const dateNumDiff = Math.round(Math.abs(dateDiffMs / oneDay));
  this.barWidth =
    (this.layout.width - this.layout.margin.left - this.layout.margin.right) / dateNumDiff;
  this.barPadding = 0.2 * this.barWidth
  this.barWidth = this.barWidth - this.barPadding
}

BarChart.prototype.addXAxis = function () {
  this.xAxis = d3.axisBottom(this.x).ticks(6).tickFormat(d3.timeFormat("%b/%d"));

  this.layerBarChart
    .append("g")
    .attr("id", "xAxis")
    .call(this.xAxis)
    .attr(
      "transform",
      `translate(${this.layout.margin.left},${this.layout.height - this.layout.margin.bottom})`
    )
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
      `translate(${this.layout.margin.left - this.barWidth / 2 - this.barPadding},${this.layout.margin.top
      })`
    )
    .style("color", this.layout.textColor)
    .call((g) => g.select(".domain").remove());
}

BarChart.prototype.addY2Axis = function () {

  this.y2Axis = d3.axisRight(this.y2).tickFormat(formatTickY2Axis);

  function formatTickY2Axis(d) {
    return this.parentNode.nextSibling ? `\xa0${d}` : `$${d}`;
  }

  this.layerBarChart
    .append("g")
    .attr("id", "y2Axis")
    .call(this.y2Axis)
    .attr(
      "transform",
      `translate(${this.layout.width - this.layout.margin.left + this.barWidth / 2 + this.barPadding
      },${this.layout.margin.top})`
    )
    .style("color", this.layout.textColor)
    .call((g) => g.select(".domain").remove());

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
