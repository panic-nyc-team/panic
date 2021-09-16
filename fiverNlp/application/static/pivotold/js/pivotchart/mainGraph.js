let MainGraph = function (pivotChart) {
    this.pivotChart = pivotChart
    this.app = pivotChart.app
    this.data = pivotChart.app.data
    this.layout = pivotChart.layout
    this.iconUrl = pivotChart.iconUrl
    this.nodes = pivotChart.nodes
    this.links = pivotChart.links
    this.fociSide = {}
    this.nodesExtras = [];
    this.clickedNode = pivotChart.clickedNode

    this.layerMain = pivotChart.layerMain
}

MainGraph.prototype.startSimulation = function () {
    let _this = this
    let layout = this.layout
    let fociSide = this.fociSide

    this.simulation.on("tick", mainTick)

    this.simulation.alphaDecay(0.001);

    function mainTick(e) {
        const nodeImageShift = (layout.nodeRadius * layout.imageNodeRatio) / 2;
        const sideImageShift = (layout.sideNodeRadius * layout.imageNodeRatio) / 2;

        _this.link
            .attr("x1", (d) => d.source.x)
            .attr("y1", (d) => d.source.y)
            .attr("x2", (d) => d.target.x)
            .attr("y2", (d) => d.target.y);

        _this.node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

        _this.nodeImage
            .attr("height", (d) =>
                d.type === "main"
                    ? _this.layout.nodeRadius * _this.layout.imageNodeRatio
                    : _this.layout.sideNodeRadius * _this.layout.imageNodeRatio
            )
            .attr("x", (d) =>
                d.type === "main" ? d.x - nodeImageShift : d.x - sideImageShift
            )
            .attr("y", (d) =>
                d.type === "main" ? d.y - nodeImageShift : d.y - sideImageShift
            );

        _this.mainHulls.attr("d", function (d) {
            return _this.hullPath(d, "main")
        });

        _this.sideHulls.attr("d", function (d) {
            return _this.hullPath(d, "extra")
        });

        _this.sideNodeText
            .attr("x", (d) => {
                if (d.type === "extra") {
                    return d.x - _this.layout.sideNodeRadius * 2;
                }
            })
            .attr("y", (d) => {
                if (d.type === "extra") {
                    return d.y - _this.layout.sideNodeRadius * 4.1;
                }
            });

        _this.sideHullsText
            .attr("width", (d) => _this.fociSide[d.cluster].clusterR)
            .attr("x", (d) => {
                return _this.fociSide[d.cluster].x - _this.fociSide[d.cluster].clusterR / 2;
            })
            .attr(
                "y",
                (d) =>
                    _this.fociSide[d.cluster].y -
                    _this.fociSide[d.cluster].clusterR -
                    _this.layout.sideNodeRadius * 2
            );
    }
}

MainGraph.prototype.addSimulation = function () {
    let _this = this
    let layout = this.layout

    this.charge = function (strength, distance) {
        return d3.forceManyBody().strength(strength).distanceMax(distance);
    };
    this.collide = function (collisionVal) {
        return d3.forceCollide().radius(collisionVal);
    };
    this.posX = function (fX, strength) {
        return d3.forceX(fX).strength(strength);
    };
    this.posY = function (fY, strength) {
        return d3.forceY(fY).strength(strength);
    };

    this.simulation = d3
        .forceSimulation(_this.pivotChart.nodes)
        .force(
            "link",
            d3
                .forceLink(_this.pivotChart.links)
                .id((d) => d.id)
                .strength(0)
        )
        .force(
            "charge",
            _this.isolateForce(
                _this.charge(-layout.nodeRadius * 1.5, layout.nodeRadius * 50),
                "main"
            )

        )
        .force("collide",
            _this.isolateForce(_this.collide(layout.nodeRadius * 1.1), "main")
        );
}

MainGraph.prototype.isolateForce = function (force, nodetype) {
    let _this = this
    let initialize = force.initialize;

    force.initialize = function () {
        initialize.call(
            force,
            _this.pivotChart.nodes.filter((node) => node.type === nodetype)
        );
    };
    return force;
}

MainGraph.prototype.getFociSide = function (extras) {
    let _this = this
    let nodes = this.pivotChart.nodes
    let hierarchyCenter = this.pivotChart.hierarchyCenter
    let distance = this.pivotChart.distance

    const mainNodesOuterRing = d3.max(
        nodes
            .filter((d) => d.type === "main")
            .map(function (d) {
                return distance(d.x - hierarchyCenter[0], d.y - hierarchyCenter[1])
            })
    );

    const newFociSide = {};
    let prevY = 0;

    this.app.extras.forEach(function (extra, i) {
        const dimensionNum = _this.getDimensions(extra).length;
        const clusterRadius =
            Math.ceil(Math.sqrt(dimensionNum)) * 2.5 * _this.layout.sideNodeRadius;
        let forceFactor = Math.log10(dimensionNum);
        forceFactor = forceFactor > 2 ? forceFactor + 1.5 : forceFactor;
        if (!Object.keys(_this.fociSide).includes(extra)) {
            obj = {
                x: _this.pivotChart.width * 0.7 * 0.5 + mainNodesOuterRing + clusterRadius,
                y: prevY + clusterRadius,
                forceFactor: forceFactor,
            };
        } else {
            obj = {
                x: _this.fociSide[extra].x,
                y: _this.fociSide[extra].y,
                forceFactor: forceFactor,
            };
        }
        prevY = obj.y + clusterRadius + 2 * _this.layout.sideNodeRadius;
        newFociSide[extra] = obj;
    });

    return newFociSide;
}

MainGraph.prototype.addTooltip = function () {
    this.tooltip = d3
        .select("body")
        .append("div")
        .attr("id", "tooltip")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("z-index", "10")
        .style("visibility", "hidden");

}


MainGraph.prototype.handleNodeClick = function (event, d) {
    let _this = this
    const dateString = new Date(d.publish_date).toDateString();
    d3.select("#tooltip").style("visibility", "visible").html(`
      <ul>
        <li class="tooltip-title">${d.title}</li>
        <li class="tooltip-date">${dateString}</li>
        <li class="tooltip-author">By: ${d.author}</li>
        <li class="tooltip-url"><a href="${d.url}" target="_blank">Source</a></li>
      </ul>
      `);
    this.tooltipPosition(event);
}

MainGraph.prototype.tooltipPosition = function (event) {
    let ttid = "#tooltip";
    let xOffset = 20;
    let yOffset = 10;
    let toolTipW = $(ttid).width();
    let toolTipeH = $(ttid).height();
    let windowY = $(window).scrollTop();
    let windowX = $(window).scrollLeft();
    let curX = event.pageX;
    let curY = event.pageY;
    let ttleft =
        curX < $(window).width() / 2
            ? curX - toolTipW - xOffset * 2
            : curX + xOffset;
    if (ttleft < windowX + xOffset) {
        ttleft = windowX + xOffset;
    } else {
        ttleft = curX - windowX - toolTipW;
    }

    let tttop =
        curY - windowY + yOffset * 2 + toolTipeH > $(window).height()
            ? curY - toolTipeH - yOffset * 2
            : curY + yOffset;
    if (tttop < windowY + yOffset) {
        tttop = curY + yOffset;
    }
    $(ttid)
        .css("top", tttop + 30 + "px")
        .css("left", ttleft + "px");
}

MainGraph.prototype.addNode = function () {
    let _this = this
    let layout = this.layout

    this.node = this.layerMain
        .append("g")
        .attr("id", "nodes")
        .attr("class", "node")
        .attr("stroke", layout.nodeStroke)
        .attr("fill", layout.nodeFill)
        .attr("stroke-width", 1.5)
        .selectAll("circle")
        .data(this.pivotChart.nodes)
        .join("circle")
        .attr("id", "mainNodes")
        .attr("r", layout.nodeRadius)
        .on("click", function (event, d) {
            if (d.type === "main") {
                _this.handleNodeClick(event, d);
                _this.node.attr("stroke", function (node) {
                    node.id === d.id ? _this.layout.linestrokeHighlight : _this.layout.nodeStroke()
                }
                );
            }
        });

    this.nodeImage = this.layerMain
        .append("g")
        .attr("id", "node-image")
        .selectAll("image");
}

MainGraph.prototype.addLink = function () {
    this.link = this.layerMain
        .append("g")
        .attr("id", "links")
        .selectAll("line");

}

MainGraph.prototype.addHulls = function () {
    let _this = this
    let layout = this.layout

    let mainHullG = this.layerMain
        .append("g")
        .attr("id", "main-hull")
        .attr("class", "hulls");

    this.mainHulls = mainHullG.selectAll("path");

    let sidehullG = this.layerMain
        .append("g")
        .attr("id", "side-hull")
        .attr("class", "side-hulls");

    this.sideHulls = sidehullG.selectAll("path");

    this.sideHullsText = this.layerMain
        .append("g")
        .attr("id", "side-hull-text")
        .selectAll("foreignObject");

    this.sideNodeText = this.layerMain
        .append("g")
        .attr("id", "side-node-text")
        .selectAll("foreignObject");
}

MainGraph.prototype.updateNodeimage = function () {
    let nodes = this.pivotChart.nodes
    let iconUrl = this.iconUrl
    let layout = this.layout


    this.nodeImage = this.nodeImage
        .data(nodes)
        .join("image")
        .style("pointer-events", "none")
        .attr("href", function (d) {
            if (d.type === "main") {
                return iconUrl.document;
            } else {
                return iconUrl[d.extra];
            }
        })
        .attr("filter", layout.imageFilter);
}


MainGraph.prototype.updateMainHulls = function () {
    let clusterMap = this.pivotChart.clusterMap
    let layout = this.layout
    let _this = this

    this.mainHulls = this.mainHulls
        .data(
            [...clusterMap].map(([k, val]) => {
                return {
                    cluster: k,
                    nodes: val.nodes,
                };
            }),
            (k) => k
        )
        .join("path")
        .attr("d", function (d) {
            return _this.hullPath(d, "main");
        })
        .attr("fill", layout.hullFill)
        .attr("opacity", layout.hullOpacity)
        .attr("stroke", layout.hullStroke)
        .attr("stroke-width", layout.hullStrokeWidth);

    this.nodeImage = this.nodeImage
        .data(this.pivotChart.nodes)
        .join("image")
        .style("pointer-events", "none")
        .attr("href", function (d) {
            if (d.type === "main") {
                return _this.iconUrl.document;
            }
            return _this.iconUrl[d.extra];
        })
        .attr("filter", this.layout.imageFilter);
}

MainGraph.prototype.clearColoring = function () {
    d3.select("#tooltip").style("visibility", "hidden");
    this.node.attr("stroke", this.layout.nodeStroke);
    this.node.attr("fill", this.layout.nodeFill);
    this.link.attr("stroke", this.layout.linestroke);
    this.clickedNode = {};
}



MainGraph.prototype.hullPath = function (data, type) {
    let layout = this.layout
    let fociSide = this.fociSide
    let clusterMap = this.pivotChart.clusterMap
    let _this = this

    let nodesPos = [];
    const nodeRadius =
        type === "main" ? layout.nodeRadius : layout.sideNodeRadius;
    const nodeRMultiplier = type === "main" ? 1.5 : 2;

    if (type === "main") {
        data.nodes.forEach((node) => {
            nodesPos = nodesPos.concat({ x: node.x, y: node.y });
        });
    } else if (type === "extra") {
        data.nodes.each((node) => {
            nodesPos = nodesPos.concat({ x: node.x, y: node.y });
        });
    }

    let cx = nodesPos.length > 0 ?
        nodesPos.map((node) => node.x).reduce((sum, x) => sum + x) /
        nodesPos.length : 0;

    let cy = nodesPos.length > 0 ?
        nodesPos.map((node) => node.y).reduce((sum, y) => sum + y) /
        nodesPos.length : 0;

    cy = type === "main" ? cy : cy - layout.sideNodeRadius;

    const maxR = d3.max(
        nodesPos.map(function (node) {

            return _this.pivotChart.distance(node.x - cx, node.y - cy)
        })
    );

    let r = maxR + nodeRadius * nodeRMultiplier;

    const p = d3.path();

    p.arc(cx, cy, r, 0, Math.PI * 2);

    if (type === "main") {
        clusterMap.get(data.cluster).cx = cx;
        clusterMap.get(data.cluster).cy = cy;
        // foci.clusterR = r;
    } else if (type === "extra") {
        fociSide[data.cluster].x = cx;
        fociSide[data.cluster].y = cy;
        fociSide[data.cluster].clusterR = nodesPos.length > 0 ? r : 0;
    }

    return nodesPos.length > 0 ? p : "M 0 0";
}

MainGraph.prototype.getDimensions = function (groupName) {
    return [...new Set(this.pivotChart.nodes.map((node) => node[groupName]))];
}

MainGraph.prototype.setNodesExtras = function () {
    let _this = this

    const oldExtras = [...new Set(
        this.pivotChart.nodes.filter((d) => d.type === "extra")
            .map((d) => d.extra)
    )]

    const removedExtra = oldExtras.filter(function (e) {
        return !_this.app.extras.includes(e)
    });

    if (removedExtra[0] !== undefined) {
        this.pivotChart.nodes = this.pivotChart.nodes
            .filter((d) => d.extra !== removedExtra[0])
        this.nodesExtras = this.nodesExtras.filter((d) => d.extra !== removedExtra[0]);
    }

    let filteredNodeExtras = this.app.extras.map(function (extra) {
        return _this.getDimensions(extra)
    }).flat()

    filteredNodeExtras = filteredNodeExtras === undefined ? [] : filteredNodeExtras

    if (filteredNodeExtras.length < this.nodesExtras.length) {
        this.nodesExtras = this.nodesExtras.filter(function (node) {
            return filteredNodeExtras.includes(node.id)
        })
    }

    const newExtras = this.app.extras.filter((e) => !oldExtras.includes(e));

    newExtras.forEach(function (extra, i) {
        obj = {};
        _this.getDimensions(extra).forEach(function (dimension, j) {
            if (!_this.nodesExtras.map((d) => d.id).includes(extra + dimension)) {
                obj = {
                    id: extra + dimension,
                    name: dimension,
                    extra: extra,
                    type: "extra",
                };
                _this.nodesExtras.push(obj);
            }
        });
    });

}

MainGraph.prototype.setMainLinks = function () {
    let _this = this

    this.pivotChart.links = this.nodesExtras
        .map(function (nodeSource) {
            const result = _this.pivotChart.nodes
                .filter(nodeTarget => nodeTarget.type === "main" && nodeTarget.type !== "extra")
                .filter(
                    (nodeTarget) => nodeSource.extra + nodeTarget[nodeSource.extra] === nodeSource.id
                )
                .map((nodeTarget) => {
                    return {
                        source: nodeSource.extra + nodeTarget[nodeSource.extra],
                        target: nodeTarget.id,
                        type: "side",
                    };
                });
            return result
        }).flat()

}

MainGraph.prototype.renderNode = function () {
    let _this = this
    this.node = this.node
        .data(_this.pivotChart.nodes, (d) => d.id)
        .join("circle")
        .on("click", function (event, d) {
            return _this.updateColoring(event, d)
        })
        .attr("fill", _this.layout.nodeFill)
        .attr("r", function (d) {
            return d.type === "main" ? _this.layout.nodeRadius : _this.layout.sideNodeRadius
        });
}

MainGraph.prototype.updateColoring = function (event, d) {
    let _this = this

    if (this.pivotChart.clickedNode === [d.id]) {
        this.node.attr("fill", this.layout.nodeFill()).attr("stroke", this.layout.nodeStroke());
    } else if (d.type === "main") {
        this.handleNodeClick(event, d);
        this.node.attr("stroke", function (node) {
            return node.id === d.id ? _this.layout.linestrokeHighlight : _this.layout.nodeStroke()
        });
    } else if (d.type === "extra") {
        const targetedNodesIds = this.pivotChart.links
            .filter((l) => l.source.id === d.id)
            .map((l) => l.target.id);

        this.link.attr("stroke", function (l) {
            let color = _this.layout.linestroke;
            if (l.source.id === d.id && l.source.id !== _this.pivotChart.clickedNode.id) {
                color = _this.layout.linestrokeHighlight;
            }
            return color;
        });

        this.node
            .attr("stroke", function (c) {
                let color = _this.layout.nodeStroke();
                if (c.id === d.id) {
                    if (d.id !== _this.pivotChart.clickedNode.id) {
                        color = _this.layout.linestrokeHighlight;
                    }
                }
                return color;
            })
            .attr("fill", function (c) {
                if (d.id !== _this.pivotChart.clickedNode.id) {
                    if (c.id !== d.id) {
                        if (targetedNodesIds.includes(c.id) && c.type === "main") {
                            return _this.layout.linestrokeHighlight;
                        }
                    }
                }
                return _this.layout.nodeFill();
            });

        this.pivotChart.clickedNode.id = this.pivotChart.clickedNode.id === d.id ? "" : d.id;
        this.pivotChart.clickedNode.type = d.type;
    }
};



MainGraph.prototype.renderLink = function () {
    this.link = this.link.data(this.pivotChart.links, (l) => [l.source, l.target]).join("line");

    this.link
        .attr("id", "link")
        .attr("stroke", this.layout.linestroke)
        .attr("stroke-width", this.layout.linestrokeWidth)
        .attr("opacity", this.layout.lineopacity);
}

MainGraph.prototype.updateSide = function () {
    let _this = this

    this.setNodesExtras()

    this.pivotChart.nodes = this.pivotChart.nodes.map((node, index) => {
        obj = node;
        if (node.type === "main") {
            obj.id = index;
        }
        return obj;
    });

    this.pivotChart.nodes = this.pivotChart.nodes
        .filter((node) => node.type === "main"
        ).concat(this.nodesExtras);

    this.fociSide = this.getFociSide(this.extras);

    this.renderNode()

    this.setMainLinks()
    this.renderLink()

    this.simulation
        .force(
            "chargeExtra",
            _this.isolateForce(
                _this.charge(-_this.layout.sideNodeRadius * 4.5, _this.layout.sideNodeRadius * 4.5),
                "extra"
            )
        )
        .force(
            "collideExtra",
            _this.isolateForce(_this.collide(_this.layout.sideNodeRadius * 3.6), "extra")
        )
        .force(
            "positionxExtra",
            _this.isolateForce(
                _this.posX(
                    function (d) {
                        return _this.fociSide[d.extra].x
                    },
                    function (d) {
                        return 0.1 / _this.fociSide[d.extra].forceFactor
                    }
                ),
                "extra"
            )
        )
        .force(
            "positionyExtra",
            _this.isolateForce(
                _this.posY(
                    function (d) {
                        return _this.fociSide[d.extra].y
                    },
                    function (d) {
                        return 0.1 / _this.fociSide[d.extra].forceFactor
                    }
                ),
                "extra"
            )
        );


    this.simulation.nodes(this.pivotChart.nodes);

    this.simulation.force("link").links(this.pivotChart.links);

    this.simulation.alpha(1).restart();

    this.nodeImage = this.nodeImage
        .data(this.pivotChart.nodes)
        .join("image")
        .style("pointer-events", "none")
        .attr("href", function (d) {
            if (d.type === "main") {
                return _this.iconUrl.document;
            } else {
                return _this.iconUrl[d.extra];
            }
        })
        .attr("filter", _this.layout.imageFilter);

    let extrasClusters = this.pivotChart.app.extras.map(function (g) {
        const obj = {
            cluster: g,
            nodes: _this.node.filter((d) => d.extra === g),
        };
        obj.counts = obj.nodes.nodes().length;
        return obj;
    });

    this.sideHulls = this.sideHulls
        .data(extrasClusters, (d) => d.cluster)
        .join("path")
        .style("cursor", "pointer")
        .attr("id", (d) => "sidehull-" + d.cluster)
        .attr("d", function (d) {
            return _this.hullPath(d, "extra")
        })
        .attr("fill", _this.layout.hullFill)
        .attr("stroke", _this.layout.hullStroke)
        .attr("stroke-width", _this.layout.hullStrokeWidth)
        .attr("opacity", _this.layout.hullOpacity)
        .call(
            d3
                .drag()
                .on("start", (e, d) => { })
                .on("drag", function (e, d) {
                    _this.simulation
                        .force(
                            "positionxExtra",
                            _this.isolateForce(
                                _this.posX(function (node) {
                                    return node.extra === d.cluster ? e.x : _this.fociSide[node.extra].x
                                }, 0.1),
                                "extra"
                            )
                        )
                        .force(
                            "positionyExtra",
                            _this.isolateForce(
                                _this.posY(function (node) {
                                    return node.extra === d.cluster ? e.y : _this.fociSide[node.extra].y
                                }, 0.1),
                                "extra"
                            )
                        );
                })
                .on("end", function (e, d) {
                    _this.fociSide[d.cluster].x = e.x;
                    _this.fociSide[d.cluster].y = e.y;
                })
        );

    //Side hull text


    this.sideHullsText = this.sideHullsText
        .data(extrasClusters)
        .join("foreignObject")
        .attr("class", "side-hull-text")
        .attr("width", function (d) {
            return _this.fociSide[d.cluster].clusterR
        })
        .attr("height", this.layout.sideNodeRadius * 2)
        .style("font-size", 20);

    d3.selectAll(".side-hull-text-div").remove();

    this.sideHullsTextSpan = this.sideHullsText
        .append("xhtml:div")
        .attr("class", "side-hull-text-div")
        .append("span")
        .attr("class", "side-text")
        .style("color", this.layout.sideFontColor)
        .html((d) => d.counts + " " + d.cluster);


    this.sideNodeText = this.sideNodeText
        .data(this.pivotChart.nodes.filter(d => d.type === "extra"))
        .join("foreignObject")
        .attr("id", (d, i) => "sideNodeText" + d.extra + i)
        .attr("class", "side-node-text")
        .style("pointer-events", "none")
        .attr("width", this.layout.sideNodeRadius * 4)
        .attr("height", this.layout.sideNodeRadius * 3)
        .style("font-size", 12);

    d3.selectAll(".sideNodeTextDiv").remove();

    this.sideNodeTextSpan = this.sideNodeText
        .append("xhtml:div")
        .attr("class", "side-node-text-div")
        .append("span")
        .attr("class", "side-text")
        .style("color", this.layout.sideFontColor)
        .html((d) => d.name);
}