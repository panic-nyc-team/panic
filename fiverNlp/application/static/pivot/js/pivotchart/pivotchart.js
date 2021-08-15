let PivotChart = function (app) {
    this.app = app
    this.data = app.data

    this.pathIcon = "/static/pivot/static/icons/";
    this.iconUrl = {}
    this.layout = {}
    this.darkMode = app.darkMode
    this.clickedNode = {};

    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.clusterMap = null;
    this.hierarchyCenter = [this.width / 2, this.height / 2];

    this.clusterMap = new Map()

    this.nodes = []
    this.links = []

    this.setLayout()
    this.setIconUrl()
    this.setNodes()

    this.draw()
}

PivotChart.prototype.addClusterMap = function (treeNodes) {
    this.clusterMap = new Map(
        treeNodes.filter((d) => d.type === "leaf").map((d) => [d.id, d])
    )
}

PivotChart.prototype.updateDarkMode = function () {
    this.bg.attr("fill", this.layout.bgColor);
    this.mainGraph.mainHulls.attr("fill", this.layout.hullFill);
    this.mainGraph.sideHulls.attr("fill", this.layout.hullFill);
    this.mainGraph.node.attr("fill", this.layout.nodeFill).attr("stroke", this.layout.nodeStroke);
    this.mainGraph.nodeImage.attr("filter", this.layout.imageFilter);
    d3.select("#document-counts").style("color", this.layout.textColor);
    d3.selectAll(".input-title").style("color", this.layout.textColor);
    d3.select(".group-by").style("background-color", this.layout.inputBg);
    d3.selectAll(".side-text").style("color", this.layout.sideFontColor);
}

PivotChart.prototype.draw = function () {
    this.layerMainBg = this.app.svg.append("g").attr("id", "layerMainBg");
    this.layerMainTransform = this.app.svg.append("g").attr("id", "initialTransform")
    this.layerMain = this.layerMainTransform.append("g").attr("id", "layerMain");
    this.layerTree = this.app.svg.append("g").attr("id", "layerTree");

    this.mainGraph = new MainGraph(this)
    this.treeGraph = new TreeGraph(this, this.mainGraph)


    this.addZoom()
    this.addInitTransform()

    this.mainGraph.addTooltip()
    this.mainGraph.addLink()
    this.treeGraph.addTree()
    this.mainGraph.addHulls()
    this.mainGraph.addNode()

    this.addBackground()

    this.mainGraph.addSimulation()
    this.treeGraph.addSimulation()

    this.treeGraph.updateTree()
    this.mainGraph.updateMainHulls()

    this.documentCounts()

    this.treeGraph.startSimulation()
    this.mainGraph.startSimulation()
}

PivotChart.prototype.restartChart = function () {
    this.setNodes()
    this.updateChart()
}

PivotChart.prototype.documentCounts = function () {
    let _this = this

    d3.select("#document-counts text").remove();
    d3.select("#document-counts")
        .append("text")
        .text(function () {
            const groupingSizes = _this.app.groupBy.map((g) => [
                g,
                new Set(_this.app.data.map((d) => d[g])).size,
            ]);
            const groupingText = groupingSizes
                .map(([g, size]) => `${size} ${g}`)
                .join(", ");
            return groupingText;
        });
}

PivotChart.prototype.addInitTransform = function () {
    let _this = this
    this.layerMainTransform.call(inittransform)

    function inittransform(g) {
        const k = 0.5 / _this.app.groupBy.length;
        g.attr(
            "transform",
            `translate(${_this.height / 2},${300 + _this.height * k}) scale(${k})`
        );
    }

}

PivotChart.prototype.updateChart = function () {
    this.treeGraph.updateTree()
    this.mainGraph.updateMainHulls()
    this.mainGraph.updateSide()
    this.mainGraph.clearColoring()
    this.documentCounts()
}

PivotChart.prototype.updateChartSide = function () {
    this.mainGraph.updateSide()
    this.mainGraph.clearColoring()
}

PivotChart.prototype.distance = function (xLength, yLength) {
    return Math.sqrt(xLength * xLength + yLength * yLength);
}

PivotChart.prototype.brighten = function (color) {
    return d3.rgb(color).brighter(0).toString();
};


PivotChart.prototype.setNodes = function () {
    this.nodes = this.app.data.map((d, index) => {
        const obj = d;
        obj.id = index;
        obj.type = "main"
        return obj;
    });
}

PivotChart.prototype.addBackground = function () {
    let _this = this
    this.bg = this.layerMainBg
        .append("g")
        .append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", this.width)
        .attr("height", this.height)
        .attr("fill", this.layout.bgColor)
        .on("click", function () {
            return _this.mainGraph.clearColoring()
        });
}

PivotChart.prototype.addZoom = function () {
    let _this = this

    this.app.svg.call(
        d3
            .zoom()
            .extent([
                [0, 0],
                [this.width, this.height],
            ])
            .scaleExtent([0, 20])
            .on("zoom", zoomed)
    );

    function zoomed({ transform }) {
        transformScale = transform.k;
        transformX = transform.x;
        transformY = transform.y;

        _this.layerMain.attr(
            "transform",
            `translate(${transform.x},${transform.y}) scale(${transform.k})`
        );
        _this.layerTree.attr(
            "transform",
            `translate(${transform.x},${transform.y}) scale(${transform.k})`
        );
    }

}


PivotChart.prototype.setIconUrl = function () {
    let pathIcon = this.pathIcon
    this.iconUrl = {
        entity: pathIcon + "entity.png",
        author: pathIcon + "author.png",
        sentiment: pathIcon + "sentiment.png",
        site: pathIcon + "site.png",
        site_type: pathIcon + "site_type.png",
        topic: pathIcon + "topic.png",
        document: pathIcon + "document.png",
        suitcase: pathIcon + "suitcase.png",
        search: pathIcon + "search.png",
    };

}

PivotChart.prototype.setLayout = function () {
    let _this = this
    this.layout = {
        nodeFill: function () { return _this.app.darkMode ? "#333" : "#fff" },
        imageFilter: function () { return _this.app.darkMode ? "invert(1)" : "invert(0)" },
        nodeRadius: 10,
        nodeStroke: function () { return _this.app.darkMode ? "#777" : "#aaa" },
        nodeStrokeWidth: 2,
        bgColor: function () { return _this.app.darkMode ? "#222" : "#fff" },
        textColor: function () { return _this.app.darkMode ? "#fff" : "#111" },
        imageNodeRatio: 1.3,
        hullFill: function () { return _this.app.darkMode ? "#222" : "#fff" },
        inputBg: function () { return _this.app.darkMode ? "#ddd" : "#fff" },
        hullStroke: "#aaa",
        hullStrokeWidth: 1,
        hullOpacity: 0.8,
        sideNodeRadius: 20,
        sideFontColor: function () { return _this.app.darkMode ? "#fff" : "#222" },
        linestroke: "#ddd",
        linestrokeWidth: 2,
        linestrokeHighlight: "#3978e6",
        lineopacity: 0.4,
        lineopacityHighlight: 1,
        treeLabelRadius: 24,
        treeRootRadius: 4,
        labelRadius: 16,
        labelCircleFill: "#eee",
        labelCircleStroke: "#eee",
        labelStrokeWidth: 2,
        labelLineStroke: "#333",
    };
}