const App = function (rawData) {
  this.rawData = rawData
  this.setDataRollup()
  this.setDataRange()
  this.setDataTree()

  this.setLayout()

  this.darkMode = true

  this.addSVGs()

  this.edgeBundling = new EdgeBundling(this);
  this.barChart = new BarChart(this)

  this.update()

  this.handleDarkMode()
}

App.prototype.update = function (params) {
  this.setDataTree()
  this.edgeBundling.render()
}

App.prototype.setDataTree = function () {
  const _this = this

  this.filteredRawData = this.rawData.filter(d => d.similarity >= _this.dataRange.start && d.similarity <= _this.dataRange.end)
  const prepared = prepareData(this.filteredRawData);
  this.treeData = prepared.tree
  this.similarityDimensions = prepared.similarityDimensions
}

App.prototype.addSVGs = function () {
  this.svgBar = d3.select("#svg-bar")
    .append("svg")
    .attr("class", ".svg-bar")
    .style("position", "absolute")
    .style("z-index", "-1")
    .style("top", 0)
    .style("left", 0)
    .style("width", 500)
    .style("height", 230)
    .style("background-color", "none");

  this.svgEdgebundling = d3
    .select("#svg-edgebundling")
    .append("svg")
    .attr("class", ".edgebundling-svg")
    .style("position", "absolute")
    .style("z-index", "-1")
    .style("top", 0)
    .style("left", 0)
    .style("width", "100%")
    .style("height", "100%")
    .style("background-color", "none");
}

App.prototype.setDataRollup = function () {
  this.dataRolled = d3.rollup(
    this.rawData,
    (v) => {
      return {
        bundleSize: v.length
      }
    },
    d => d.similarity
  )
}

App.prototype.handleDarkMode = function () {
  const _this = this
  const toggleDark = d3.select("#toggle-dark");
  const localStorage = window.localStorage;

  if (localStorage.edgeBundlingDarkMode) {
    const valString = localStorage.edgeBundlingDarkMode
    if (valString === "true") {
      this.darkMode = true
    } else {
      this.darkMode = false
    }
  }

  toggleDark.node().checked = this.darkMode;
  this.updateLayout();

  toggleDark.on("click", function (e) {
    _this.darkMode = this.checked;
    window.localStorage.edgeBundlingDarkMode = _this.darkMode;
    _this.updateLayout();
  });
}

App.prototype.updateLayout = function (params) {
  this.edgeBundling.setColor()
}

App.prototype.setLayout = function () {
  let _this = this
  this.props = {
    linkBaseColor: "#aaa",
    linkWidth: 1,
    linkWidthHighlight: 3,
    nodeColor: () => (_this.darkMode ? "#eee" : "#222"),
    nodeFontSize: 10,
    nodeFontSizeBold: 16,
    nodeMargin: 2,
    inputBgColor: "#ccc",
    controlBoxBg: () => (_this.darkMode ? "#666" : "#fff"),
    controlBoxColor: () => (_this.darkMode ? "#fff" : "#111"),
    controlBoxColor2: () => (_this.darkMode ? "#444" : "#eee"),
    inputBgAll: "#444",
    colorHighlight: "red",
    windowHeight: window.innerHeight,
    windowWidth: window.innerWidth,
    arcWidth: 5,
    arcMargin: 0,
    bgColor: () => (_this.darkMode ? "#222" : "#fff"),
    groupLabelSize: 22,
    groupLabelRatio: 0.45,
    groupLinesColor: () => _this.darkMode ? "#fff" : "#111",
    groupLabelOpacity: 0.4,
    tooltipBg: () => (_this.darkMode ? "#ddd" : "#fff"),
    textEstimateL: 200,
  };

}

App.prototype.setDataRange = function () {
  this.dataRange = {
    start: d3.min(this.dataRolled.keys()),
    end: d3.max(this.dataRolled.keys())
  }
}

d3.json(report_name)
  .then(function (rawData) {
    const app = new App(rawData)
  }).catch(function (error) {
    console.log(error);
  });