const EdgeBundling = function (app) {
    this.app = app
    this.setUp()
    this.addLayers()
    this.init()
}

EdgeBundling.prototype.setUp = function () {
    const _this = this
    this.groupBy = "parent_site";

    this.dimension = "all"

    this.linkColors = d3.scaleOrdinal().range(["#d94040", "#397fda", "#0099aa", "#b5904b", "#b518bb", "#99b93f", "#299f60", "#b35222"]
    );

    this.darken = function (color) {
        return _this.app.darkMode
            ? d3.rgb(color).brighter(3).toString()
            : d3.rgb(color).darker(2).toString();
    };

    this.deltaRad = Math.PI / 2

    this.prevWheeled = ""

    this.nodesNumber = this.app.treeData.children
        .map((d) => d.children.length)
        .reduce((sum, x) => sum + x);

    this.radius = this.getRadius()

    this.controlBoxHeight = d3.select(".control").node().getBoundingClientRect()
        .height;

    this.line = d3.lineRadial().curve(d3.curveBundle.beta(0.85))
        .radius((d) => d.y - this.app.props.arcWidth - this.app.props.arcMargin)
        .angle((d) => d.x);
}


EdgeBundling.prototype.setDimensions = function (k) {
    this.dimension = k
}

EdgeBundling.prototype.addLayers = function () {
    const _this = this
    this.layerBg = this.app.svgEdgebundling.append("g").attr("class", "bg");
    this.layerChart = this.app.svgEdgebundling.append("g").attr("class", "chart").style("cursor", "pointer")
    this.layerEdgeBundling = this.layerChart.append("g").attr("class", "edgebundling").call(transformInit)
    this.layerWheel = this.layerEdgeBundling.append("g").attr("class", "wheel")
    this.layerNodes = this.layerEdgeBundling.append("g").attr("class", "nodes");
    this.layerLinks = this.layerEdgeBundling.append("g").attr("class", "links");
    this.layerGroupLabel = this.layerEdgeBundling.append("g").attr("class", "arcs");

    function transformInit(g) {
        const k =
            (_this.app.props.windowHeight - _this.controlBoxHeight) / 2 / (_this.radius + _this.app.props.textEstimateL);
        const x = _this.app.props.windowWidth / 2
        const y = (_this.radius + _this.app.props.textEstimateL * 2) * k
        g.attr(
            "transform",
            `translate(${x},${y}) scale(${k})`
        );
    }
}

EdgeBundling.prototype.init = function () {
    this.addBg()
    this.addTooltip()
    this.addAudio()
    this.addWheel()
    this.addDocumentCounts()
    this.addController()

    this.addZoomEvent()
    this.addWheelEvent()
}

EdgeBundling.prototype.getRadius = function () {
    const r = ((this.nodesNumber + this.app.treeData.children.length) * (this.app.props.nodeFontSize + this.app.props.nodeMargin)) / (2 * Math.PI);
    return r > 300 ? r : 300


}

EdgeBundling.prototype.render = function () {
    d3.selectAll(".dimension").classed("active", false);
    d3.select(`#dimension-${this.dimension}`).classed("active", true);

    this.nodesNumber = this.app.treeData.children
        .map((d) => d.children.length)

    if (this.nodesNumber.length > 0) {
        this.nodesNumber = this.nodesNumber
            .reduce((sum, x) => sum + x);

        this.radius = this.getRadius()
        this.tree = d3.cluster().size([2 * Math.PI, this.radius]);
        this.root = this.tree(bilink(d3.hierarchy(this.app.treeData)));

        this.addWheel()
        this.addNode()
        this.addLink()
        this.addGroupLabel()

        this.setColor()
        this.inputsUpdate();
        this.updateLink();
    } else {
        d3.selectAll(".node-g").remove()
        d3.selectAll(".link-path").remove()
        d3.selectAll(".group-g").remove()
    }

    this.addDocumentCounts()

}

EdgeBundling.prototype.addBg = function () {
    this.bg = this.layerBg
        .append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", this.app.props.windowWidth)
        .attr("height", this.app.props.windowHeight)
        .attr("fill", this.app.props.bgColor)
        .on("click", clickBg);

    function clickBg(event, d) {
        $(".node-text").d3Mouseout()
        d3.select("#tooltip").style("visibility", "hidden");
    }
}

EdgeBundling.prototype.addWheel = function () {
    d3.select(".wheel-path").remove()
    d3.select(".wheel-needle").remove()
    this.wheel = this.layerWheel
        .append("path")
        .attr("class", "wheel-path")
        .attr("d", d3.arc()
            .innerRadius(this.radius)
            .outerRadius(this.radius + this.app.props.textEstimateL * 1.5)
            .startAngle(0)
            .endAngle(2 * Math.PI))

    this.wheelNeedle = this.layerWheel.append("rect")
        .attr("class", "wheel-needle")
        .attr("x", this.radius)
        .attr("y", -this.app.props.nodeFontSize * 1.5)
        .attr("width", this.app.props.textEstimateL)
        .attr("height", this.app.props.nodeFontSize * 2)
        .attr("fill", this.app.props.groupLinesColor)
        .attr("opacity", 0.1)
}

EdgeBundling.prototype.addDocumentCounts = function (params) {
    const documentCounts = d3.select("#document-counts").html(this.app.filteredRawData.length);
}

EdgeBundling.prototype.addController = function (params) {
    const _this = this

    this.inputsDimension = d3
        .select("#similarity-dimension")
        .selectAll("div")
        .data(this.app.similarityDimensions)
        .join("div")
        .style("background-color", this.app.props.inputBgColor)
        .attr("class", "button");

    this.inputsDimensionText = this.inputsDimension
        .append("span")
        .style("background-color", "transparent")
        .attr("class", "noselect")
        .attr("id", (k) => k)
        .html((k) => k)
        .attr("id", (k) => "dimension-" + k)
        .attr("class", (k) => "dimension")
        .attr("value", (k) => k)
        .on("click", handleInput);

    this.inputsGroupBy = d3
        .select("#group-by")
        .append("div")
        .attr("class", "button")
        .style("background-color", this.app.props.inputBgColor)
        .html(this.groupBy);

    function handleInput(event, k) {
        d3.selectAll(".dimension").classed("active", false);
        d3.select(`#dimension-${k}`).classed("active", true);

        _this.setDimensions(k);
        _this.inputsUpdate();
        _this.updateLink();
    }
}

EdgeBundling.prototype.inputsUpdate = function () {
    const _this = this
    this.inputsDimension.style("background-color", (k) => {
        return _this.dimension === k ? _this.linkColors(k) : _this.app.props.inputBgColor;
    });
}

EdgeBundling.prototype.textCanvas = function (text) {
    const textLength = text.length * this.app.props.nodeFontSize * 0.53
    const w = textLength
    const h = this.app.props.nodeFontSize

    const foreignObject = this.layerEdgeBundling.append("g")
        .append("foreignObject")
        .attr("x", 0)
        .attr("y", 0)
        .attr("height", h * 2)
        .attr("width", w);

    const canvas = foreignObject.append("xhtml:canvas")
        .attr("x", 0)
        .attr("y", 0)
        .attr("height", h * 2)
        .attr("width", w)

    const ctx = canvas.node().getContext("2d")
    ctx.clearRect(0, 0, textLength, this.app.props.nodeFontSize)
    ctx.font = `${h}px Gotham`;
    ctx.fillStyle = "black"
    ctx.fillText(text, 0, h)
}

EdgeBundling.prototype.getTextWidth = function (text) {
    return text.length * this.app.props.nodeFontSize * 0.53
}

EdgeBundling.prototype.appendTextCanvas = function (that, g) {
    const _this = this
    const font = that.app.props.nodeFontSize
    const foH = font * 1.2
    const dpr = window.devicePixelRatio || 1;
    that.nodeTextFo = g.append("foreignObject")
        .attr("class", "node-foreignobject")
        .attr("x", d => Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? 0 : -that.getTextWidth(that.excerpt(d.data.text)))
        .attr("y", -foH * 3 / 4)
        .attr("height", 1)
        .attr("overflow", "visible")
        .attr("width", d => that.getTextWidth(that.excerpt(d.data.text)))
        .attr("transform", (d) => (Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? "rotate(0)" : "rotate(180)"))

    that.nodeTextCanvas = that.nodeTextFo
        .append('xhtml:canvas')
        .attr("class", "node-canvas")
        .attr("height", foH * dpr)
        .attr("width", d => that.getTextWidth(that.excerpt(d.data.text)) * dpr)
        .each(drawCtx)
        .on("click", function (event, d) {
            _this.nodeClicked(this, event, d)
        })
        .on("mouseover",
            function (event, d) {
                _this.nodeOvered(this, event, d)
            })
        .on("mouseout", function (event, d) {
            _this.nodeOuted(this, event, d)
        })


    function drawCtx(d) {
        const ctx = this.getContext('2d')
        ctx.scale(dpr, dpr)
        ctx.font = `${font}px Gotham`
        ctx.fillStyle = that.app.props.nodeColor()
        ctx.fillText(d.data.text, 0, font)
    }
}


EdgeBundling.prototype.addNode = function () {
    const _this = this
    const data = this.root.leaves()

    d3.selectAll(".node-g").remove()
    d3.selectAll(".node-text").remove()


    this.node = this.layerNodes
        // .attr("font-family", "sans-serif")
        // .attr("font-size", this.app.props.nodeFontSize)
        .selectAll("g")
        .data(data, d => d)
        .join((enter) => enter
            .append("g")
            .attr("id", d => "g-node" + `${d.data.id}`)
            .attr("class", "node-g")
            .attr("transform", (d) => {
                return `rotate(${d.x / Math.PI * 180 - 90}) translate(${d.y},0)`
            })
        )
        .call(function (g) {
            _this.appendTextCanvas(_this, g)
        })

    // .append("text")
    // .attr("class", "node-text")
    // .attr("id", d => "node" + `${d.data.id}`)
    // .style("font-family", d => "Gotham")
    // .attr("stroke", this.app.props.nodeColor)
    // .style("cursor", "pointer")
    // .style("pointer-events", "click")
    // .attr("dy", "0.31em")
    // .attr("x", (d) => (Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? 6 : -6))
    // .attr("text-anchor", (d) => (Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? "start" : "end"))
    // .attr("transform", (d) => (Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? "rotate(0)" : "rotate(180)"))
    // .text(function (d) { return _this.excerpt(d.data.text) })
    // .each(function (d) {
    //     d.text = this;
    //     d.group = d.data.group;
    // })
    // .on("click", function (event, d) {
    //     _this.nodeClicked(this, event, d)
    // })
    // .on("mouseover",
    //     function (event, d) {
    //         _this.nodeOvered(this, event, d)
    //     })
    // .on("mouseout", function (event, d) {
    //     _this.nodeOuted(this, event, d)
    // })

    this.layerNodes.attr("transform", `rotate(${_this.deltaRad / Math.PI * 180})`)
}

EdgeBundling.prototype.nodeOvered = function (that, event, d) {
    const _this = this
    _this.link.style("mix-blend-mode", null);
    d3.select(that).attr("font-weight", "bold")
        .attr("font-size", _this.app.props.nodeFontSizeBold)
        .text(_this.excerptBold(d.data.text))

    let pairedNodes = d.outgoing.concat(d.incoming)
    // .filter(function (k) {
    //     return _this.dimensionAll
    //         ? k.similarity_dimension === "all"
    //         : k.similarity_dimension !== "all";
    // });


    d3.selectAll(
        pairedNodes.map((k) => {
            return k.path;
        })
    )
        .attr("stroke-width", _this.app.props.linkWidthHighlight)
        .attr("stroke", (k) => {
            return _this.dimension === k.similarity_dimension
                ? _this.darken(_this.linkColors(k.similarity_dimension))
                : _this.darken(_this.app.props.linkBaseColor);
        })
        .raise();

    const targetText = new Map();

    d3.selectAll(
        pairedNodes.map((k) => {
            const [i, o] = k;
            const selectedId = i.data.id === d.data.id ? o.data.id : i.data.id;
            targetText.set(selectedId, {
                similarity_dimension: k.similarity_dimension,
                similarity: k.similarity,
            });
            return i.data.id === d.data.id ? o.text : i.text;
        })
    )
        .attr("font-weight", "bold")
        .attr("fill", (k) => {
            const val = targetText.get(k.data.id).similarity_dimension;
            return _this.dimension === val ? _this.linkColors(val) : _this.app.props.nodeColor();
        });
}

EdgeBundling.prototype.nodeOuted = function (that, event, d) {
    const _this = this
    const pairedNodes = d.outgoing.concat(d.incoming);

    _this.link.style("mix-blend-mode", null);
    d3.select(that).attr("font-weight", null)
        .attr("fill", _this.app.props.nodeColor)
        .attr("font-size", _this.app.props.nodeFontSize)
        .text(_this.excerpt(d.data.text));
    d3.selectAll(pairedNodes.map((d) => d.path))
        .attr("stroke", (d) => {
            return _this.dimension === d.similarity_dimension
                ? _this.linkColors(d.similarity_dimension)
                : _this.app.props.linkBaseColor;
        })
        .attr("stroke-width", _this.app.props.linkWidth);
    d3.selectAll(
        pairedNodes.map(([i, o]) => (i.data.id === d.data.id ? o.text : i.text))
    )
        .attr("fill", _this.app.props.nodeColor)
        .attr("font-weight", null);
}

EdgeBundling.prototype.nodeClicked = function (that, event, d) {
    const _this = this
    const pairedNodes = d.outgoing.concat(d.incoming);
    const targetText = new Map();

    const targets = pairedNodes.map((k) => {
        const [i, o] = k;
        const selected = i.data.id === d.data.id ? o.data : i.data;
        targetText.set(k.similarity_dimension + selected.id, {
            similarity_dimension: k.similarity_dimension,
            similarity: k.similarity,
        });
        return i.data.id === d.data.id
            ? { node: o, group: k.similarity_dimension }
            : { node: i, group: k.similarity_dimension };
    });

    d3.select("#tooltip").style("visibility", "visible").html(`
        <h4 class="tooltip-title">${d.data.group}</h4>
        <p class="tooltip-text"><a class="tooltip-url" href="${d.data.group_url
        }" target="_blank">${d.data.text}</a></p>
        <h4>VS</h4>
        <ul class="tooltip-list">
          ${targets
            .sort((a, b) => {
                const la = targetText.get(a.group + a.node.data.id);
                const lb = targetText.get(b.group + b.node.data.id);
                return lb.similarity - la.similarity;
            })
            .map((k) => {
                const val = targetText.get(k.group + k.node.data.id);
                const data = k.node.data;
                return `
              <li >
                <a class="tooltip-url"
                   href="${data.group_url}" 
                   target="_blank">
                  <div class="tooltip-list-element">
                    <div style="background-color:${_this.linkColors(
                    val.similarity_dimension
                )};">
                      <p style="color:white;">${val.similarity}%</p>
                    </div>
                    <p><span class="tooltip-element-title"><strong>${data.group}</strong></span><br>
                        <span class="tooltip-list-element-text">${data.text
                        .split(" ")
                        .slice(0, 5)
                        .join(" ")}...</span>
                    </p>
                  </div>
                </a>
              </li>
              `;
            })
            .join("")}
        </ul>
      `);

    this.tooltipPosition(event);
}

EdgeBundling.prototype.addLink = function () {
    const _this = this
    d3.selectAll(".link-path").remove()
    this.link = this.layerLinks
        .style("pointer-events", "none")
        .attr("stroke", this.app.props.linkBaseColor)
        .attr("stroke-width", this.app.props.linkWidth)
        .attr("fill", "none")
        .selectAll("path")
        .data(this.root.leaves().flatMap((leaf) => leaf.outgoing), d => d)
        .join(
            (enter) => enter.append("path")
                .attr("class", "link-path")
                .attr("d", ([i, o]) => {
                    return _this.line(i.path(o));
                })
                .each(function (d) {
                    this.similarity = d.similarity;
                    this.similarity_dimension = d.similarity_dimension;
                    d.path = this;
                })
                .attr("opacity", function (d) {
                    return this.similarity / 100;
                })
        )
    this.layerLinks.attr("transform", `rotate(${_this.deltaRad / Math.PI * 180})`)


}


EdgeBundling.prototype.addGroupLabel = function () {
    const _this = this
    for (const leaves of this.root.children) {
        const childrensX = leaves.children.map((c) => c.x);
        leaves.groupName = leaves.data.groupName;
        leaves.arcX = {
            start: d3.min(childrensX),
            end: d3.max(childrensX),
        };
    }

    const data = this.root.children.map(d => {
        const obj = d.arcX
        obj.groupName = d.groupName
        return obj
    })

    d3.selectAll(".group-g").remove()

    this.groupG = this.layerGroupLabel
        .selectAll("g")
        .data(data, d => d)
        .join(enter => enter.append("g")
            .attr("class", "group-g"),
        )

    this.groupLabelLines = this.groupG
        .append("path")
        .style("cursor", "pointer")
        .attr("id", (d) => ("group-label" + d.groupName).replace(/\s/g, ""))
        .attr("class", "group-label")
        .attr("stroke", this.app.props.groupLinesColor)
        .attr("fill", "none")
        .attr("opacity", this.app.props.groupLabelOpacity)
        .attr("d", (d) => {
            const radian = //d.start + (d.end - d.start) / 2
                _this.newRad(_this.deltaRad, d)
            return _this.drawLabelLines(radian, d.start, d.end, d.groupName)
        })

    this.groupLabelArc = this.groupG
        .append("path")
        .attr("id", (d) => ("arc" + d.groupName).replace(/\s/g, ""))
        .attr("fill", (d, i) => {
            return i === 0 ? _this.linkColors(d.similarity_dimension) : _this.app.props.groupLinesColor
        })
        .attr("opacity", (d, i) => {
            return i === 0 ? 1 : _this.app.props.groupLabelOpacity
        })
        .call(callDrawLabelArc)
        .attr("transform", () => `rotate(${_this.deltaRad / (Math.PI * 2) * 360})`)

    function callDrawLabelArc(g) {
        _this.drawLabelArc(g)
    }

    this.groupLabelBg = this.groupG
        .append("path")
        .attr("fill", this.app.props.bgColor)
        .attr("stroke", "none")
        .attr("d", (d) => {
            const radian = //d.start + (d.end - d.start) / 2
                _this.newRad(_this.deltaRad, d)
            return _this.drawLabelBg(radian, d.start, d.end, d.groupName)
        })

    this.groupText = this.groupG
        .append("text")
        // .style("pointer-events", "none")
        .style("font-family", "Gotham")
        .attr("rotate", (d) =>
            _this.newRad(_this.deltaRad, d) > Math.PI ? "180" : "0")

    this.groupTextPath = this.groupText
        .append("textPath")
        .attr("xlink:href", (d) => ("#group-label" + d.groupName).replace(/\s/g, ""))
        .style("font-size", this.app.props.groupLabelSize)
        .style("text-anchor", "end")
        .style("alignment-baseline", (d) => "middle")
        .attr("fill", "white")
        .text((d) => {
            return _this.newRad(_this.deltaRad, d) > Math.PI
                ? d.groupName.split("").reverse().join("")
                : d.groupName
        })
        .attr("startOffset", "100%")


}

EdgeBundling.prototype.addZoomEvent = function () {
    const _this = this
    const throttled = _.throttle(zoomed, 50)
    const zoom =
        d3
            .zoom()
            .extent([
                [0, 0],
                [this.app.props.windowWidth, this.app.props.windowHeight],
            ])
            .scaleExtent([0, 20])
            .on("zoom", throttled)

    let zoomedElement = this.layerBg.call(zoom);

    d3.select("#toggle-center-button").on('click', function () {
        zoomedElement.transition()
            .duration(750).call(zoom.transform, d3.zoomIdentity);
    })

    function zoomed({ transform }) {
        _this.layerChart
            .attr(
                "transform",
                `translate(${transform.x},
           ${transform.y}) 
                  scale(${transform.k}) `
            );
    }
}


EdgeBundling.prototype.addWheelEvent = function () {
    const _this = this
    const throttled = _.throttle(wheeled, 10)
    this.layerChart.call(
        d3.zoom()
            .extent([
                [0, 0],
                [_this.app.props.windowWidth, _this.app.props.windowHeight],
            ])
            .scaleExtent([0, Infinity])
            .on("zoom", throttled)
    )

    function callWheeled(e) {
        _this.wheeled(e)
    }

    function wheeled({ transform, sourceEvent }) {
        const nodeRad = _this.app.props.nodeFontSize / _this.radius
        const rotationY = Math.abs(Math.floor(sourceEvent.wheelDeltaY / 120))

        rotateWheel()

        function rotateWheel() {
            const e = sourceEvent.type == "mousemove" ? sourceEvent.movementY :
                sourceEvent.type == "wheel" ? sourceEvent.deltaY : 0

            const assumedSegment = ((_this.root.leaves().length + _this.root.children.length)) * 2 + 1

            _this.deltaRad = e > 0 ? _this.deltaRad + 2 * Math.PI / assumedSegment : _this.deltaRad - 2 * Math.PI / assumedSegment

            _this.layerLinks.attr("transform", (d) => `rotate(${((_this.deltaRad) / Math.PI) * 180}) `)

            _this.layerNodes.attr("transform", (d) => `rotate(${((_this.deltaRad) / Math.PI) * 180}) `)

            // _this.node
            //     .attr("x", (d) => (Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? 6 : -6))
            //     .attr("text-anchor", (d) => (Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? "start" : "end"))
            //     .attr("transform", (d) => (Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? "rotate(0)" : "rotate(180)"))

            _this.nodeTextFo
                .attr("x", d => Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? 0 : -_this.getTextWidth(_this.excerpt(d.data.text)))
                .attr("transform", (d) => (Math.sin(d.x + _this.deltaRad) > Math.sin(Math.PI) ? "rotate(0)" : "rotate(180)"))

            _this.groupLabelLines.attr("d", (d, i) => {
                return _this.drawLabelLines(_this.newRad(_this.deltaRad, d), d.start, d.end, d.groupName)
            })

            _this.groupLabelArc.attr("transform", () => `rotate(${_this.deltaRad / (Math.PI * 2) * 360})`)

            _this.groupLabelBg
                .attr("d", (d) => {
                    return _this.drawLabelBg(_this.newRad(_this.deltaRad, d), d.start, d.end, d.groupName)
                })

            _this.groupText
                .attr("rotate", (d) => {
                    return _this.newRad(_this.deltaRad, d) > Math.PI ? "180" : "0"
                })

            _this.groupTextPath
                .text((d) => {
                    return _this.newRad(_this.deltaRad, d) > Math.PI
                        ? d.groupName.split("").reverse().join("")
                        : d.groupName
                })


            // Fake click and hover event triggering tooltip on wheel

            const nodeFocus = _this.root.leaves()
                .filter(d => {
                    let radianFocus = (d.x + _this.deltaRad) % (Math.PI * 2)
                    radianFocus = radianFocus < 0 ? 2 * Math.PI + radianFocus : radianFocus
                    return radianFocus > Math.PI / 2 - _this.app.props.nodeFontSize / _this.radius &&
                        radianFocus < Math.PI / 2 + _this.app.props.nodeFontSize / _this.radius
                })[0]

            // if (nodeFocus !== undefined) {
            //     const fakeClickEvent = { pageX: sourceEvent.clientX + _this.props.textEstimateL * 2 + 100, pageY: controlBoxHeight + 30 }
            //     _this.nodeClicked(fakeClickEvent, nodeFocus)

            //     $(".node-text").d3Mouseout()
            //     $(`#node${nodeFocus.data.id}`).d3Mouseover()

            //     // if (prevWheeled !== nodeFocus.data.id) {
            //     //     $(`#audio-wheel-button`).d3Mouseclick()
            //     //     prevWheeled = nodeFocus.data.id
            //     // }

            // }
        }
    }
}

EdgeBundling.prototype.newRad = function (deltaRad, d) {
    let radian = ((d.start + (d.end - d.start) / 2) + deltaRad) % (Math.PI * 2)

    radian = radian > 0 ? radian : 2 * Math.PI + radian
    return radian
}

EdgeBundling.prototype.addAudio = function () {
    const _this = this
    this.enableSoundNotice = d3.select(".toggle-sound-notice")
        .style("background-color", this.app.props.groupLinesColor)
        .style("color", this.app.props.bgColor)

    this.app.svgEdgebundling.on("click", () => {
        _this.enableSoundNotice.style("display", "none")
    })

    const audioPromise = document.getElementById("audio-wheel")

    d3.select("#audio-wheel-button").on("click", () => {
        audioPromise.currentTime = 0
        audioPromise.play()
    })

}


EdgeBundling.prototype.addTooltip = function () {
    this.tooltip = d3
        .select("body")
        .append("div")
        .attr("id", "tooltip")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("z-index", "10")
        .style("visibility", "hidden");
}

EdgeBundling.prototype.handleLabelHeight = function (deg, groupName) {

    const outer = this.radius + this.app.props.textEstimateL * 1.2 + this.app.props.arcWidth

    let assumedLeaves = this.root.leaves().length + this.root.children.length
    assumedLeaves = assumedLeaves < 200 ? 200 : assumedLeaves < 300 ? 300 : assumedLeaves

    let factor = ((deg / (Math.PI * 2) * assumedLeaves) % (assumedLeaves / 2)) - (assumedLeaves / 4)
    factor = deg >= Math.PI && deg < Math.PI * 2 ? factor : -factor

    let groupToNodeRatio = (this.root.children.length / this.root.leaves().length) * 0.5

    const y2 = -factor * this.app.props.nodeFontSize * (1.5 + groupToNodeRatio)

    let textLen =
        groupName.length * this.app.props.groupLabelSize * this.app.props.groupLabelRatio;
    textLen = deg > Math.PI ? -textLen : textLen

    return { assumedLeaves: assumedLeaves, outer: outer, y2: y2, textLen: textLen }
}

EdgeBundling.prototype.drawLabelLines = function (deg, start, end, groupName) {
    let { outer, assumedLeaves, y2, textLen } = this.handleLabelHeight(deg, groupName)
    return d3.line()([
        // [Math.sin(deg)*radius,-Math.cos(deg)*radius],
        [Math.sin(deg) * outer, -Math.cos(deg) * outer],
        [Math.sin(deg) * (outer + 100), y2],
        [Math.sin(deg) * (outer + 100) + textLen * 1.3, y2],
    ])
}

EdgeBundling.prototype.drawLabelBg = function (deg, start, end, groupName) {
    let { outer, assumedLeaves, y2, textLen } = this.handleLabelHeight(deg, groupName)

    const halfFont = this.app.props.groupLabelSize / 2

    return d3.line()([
        [Math.sin(deg) * (outer + 100) + textLen * 0.05, y2 - halfFont],
        [Math.sin(deg) * (outer + 100) + textLen * 0.05, y2 + halfFont],
        [Math.sin(deg) * (outer + 100) + textLen * 1.3, y2 + halfFont],
        [Math.sin(deg) * (outer + 100) + textLen * 1.3, y2 - halfFont],
    ])
}

EdgeBundling.prototype.drawLabelArc = function (g) {
    const _this = this
    const radNode = this.app.props.nodeFontSize / (this.radius + this.app.props.textEstimateL) / 2
    const arc =
        d3.arc()
            .innerRadius(this.radius + this.app.props.textEstimateL * 1.2)
            .outerRadius((d, i) => {
                const extension = i === 0 ? 2 : 0
                return _this.radius + _this.app.props.textEstimateL * 1.2 + _this.app.props.arcWidth + extension
            })
            .startAngle((d) => {
                const start = d.end - d.start == 0 ? d.start - radNode : d.start
                const end = d.end - d.start == 0 ? d.end + radNode : d.end
                return start > Math.PI / 2 && start < Math.PI * 1.5 ? end : start;
            })
            .endAngle((d) => {
                const start = d.end - d.start == 0 ? d.start - radNode : d.start
                const end = d.end - d.start == 0 ? d.end + radNode : d.end
                return start > Math.PI / 2 && start < Math.PI * 1.5 ? start : end;
            });

    g.attr("d", arc)
}


EdgeBundling.prototype.setColor = function () {
    const _this = this

    this.bg.attr("fill", this.app.props.bgColor);
    // d3.selectAll(".node-text").attr("stroke", this.app.props.nodeColor);
    this.addNode()
    this.tooltip
        .style("background-color", this.app.props.tooltipBg)
        .style("color", this.app.props.tooltipBg);

    d3.select(".document-counts")
        .style("background-color", "transparent")
        .style("color", this.app.props.controlBoxColor);

    d3.selectAll(".inputs-bg").style("background-color", "transparent");
    d3.selectAll(".inputs-bg h4").style("color", this.app.props.controlBoxColor);
    d3.selectAll(".group-by")
        .style("background-color", this.app.props.controlBoxColor2)
        .style("border", "none");
    d3.selectAll(".similarity-dimension")
        .style("background-color", this.app.props.controlBoxColor2)
        .style("border", "none");

    this.wheel.attr("fill", this.app.props.bgColor)

    this.wheelNeedle.attr("fill", this.app.props.groupLinesColor)

    this.groupLabelBg.attr("fill", this.app.props.bgColor)

    this.groupLabelLines.attr("stroke", this.app.props.groupLinesColor)

    this.groupTextPath.attr("fill", this.app.props.groupLinesColor)

    this.groupLabelArc
        .attr("fill", this.app.props.groupLinesColor)

    this.enableSoundNotice
        .style("background-color", this.app.props.groupLinesColor)
        .style("color", this.app.props.bgColor)
}


EdgeBundling.prototype.updateLink = function () {
    const _this = this
    this.link
        .attr("stroke", function (d) {
            return _this.dimension === d.similarity_dimension
                ? _this.linkColors(d.similarity_dimension)
                : _this.app.props.linkBaseColor;
        })
        .attr("opacity", function (d) {
            return _this.dimension === d.similarity_dimension
                ? (d.similarity / 100).toFixed(2)
                : 0;
        });
}


EdgeBundling.prototype.excerpt = function (text) {
    text = text.split(" ").slice(0, 5).join(" ")
    const textLength = text.length * this.app.props.nodeFontSize
    return textLength > 400 ? text.slice(0, Math.floor(400 / this.app.props.nodeFontSize)) : text;
}

EdgeBundling.prototype.excerptBold = function (text) {
    text = text.split(" ").slice(0, 5).join(" ")
    const textLength = text.length * this.app.props.nodeFontSizeBold
    return textLength > 400 ? text.slice(0, Math.floor(400 * 0.9 / this.app.props.nodeFontSizeBold)) : text;
}



EdgeBundling.prototype.tooltipPosition = function (event) {
    let ttid = "#tooltip";
    let xOffset = 10;
    let yOffset = 10;
    let toolTipW = $(ttid).width();
    let toolTipeH = $(ttid).height();
    let windowY = $(window).scrollTop();
    let windowX = $(window).scrollLeft();
    let curX = event.pageX;
    let curY = event.pageY;

    let ttleft =
        curX < $(window).width() / 2
            ? curX - toolTipW - xOffset * 3
            : curX + xOffset;

    if (ttleft < windowX + xOffset) {
        ttleft = windowX + xOffset;
    } else {
        ttleft = curX - windowX - toolTipW;
    }

    let tttop =
        curY + toolTipeH + yOffset * 3 > $(window).height()
            ? curY - toolTipeH - yOffset * 3
            : curY + yOffset;

    if (tttop < windowY + yOffset) {
        tttop = curY + yOffset;
    }

    $(ttid)
        .css("top", tttop + 20 + "px")

        .css("left", ttleft + "px");
}

// Simulating mouse click event
jQuery.fn.d3Mouseclick = function () {
    this.each(function (i, e) {
        var evt = new MouseEvent("click");
        e.dispatchEvent(evt);
    });
};


// Simulating mouseover event
jQuery.fn.d3Mouseover = function () {
    this.each(function (i, e) {
        var evt = new MouseEvent("mouseover");
        e.dispatchEvent(evt);
    });
};

// Simulating mouseout event
jQuery.fn.d3Mouseout = function () {
    this.each(function (i, e) {
        var evt = new MouseEvent("mouseout");
        e.dispatchEvent(evt);
    });

};

function bilink(root) {
    const map = new Map(root.leaves().map((d) => [d.data.id, d]));
    for (const d of root.leaves()) {
        d.incoming = [];
        d.outgoing = d.data.targets.map((i) => {
            const result = [d, map.get(i.id)];
            result.similarity = i.similarity;
            result.similarity_dimension = i.similarity_dimension;
            return result;
        });
    }
    for (const d of root.leaves())
        for (const o of d.outgoing) o[1].incoming.push(o);

    // for (const d of root.leaves()) d.outgoing = d.outgoing.concat(d.incoming);


    return root;
}
