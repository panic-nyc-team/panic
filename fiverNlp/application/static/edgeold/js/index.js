d3.json(report_name)
  .then(function (rawData) {
    console.log(report_name);
    const { tree, similarityDimensions } = prepareData(rawData);
    EdgeBundling(rawData, tree, similarityDimensions);
  }).catch(function (error) {
    console.log(error);
  });

function EdgeBundling(rawData, treeData, rawSimilarityDimensions) {
  let groupBy = "parent_site";
  let similarityDimensions = rawSimilarityDimensions;

  let dimensions = [];
  let dimensionAll = false;

  const setGroupBy = (value) => {
    groupBy = value;
  };

  const setDimensions = (k) => {
    dimensions = [k];
  };

  const linkColors = d3.scaleOrdinal().range(d3.schemeSet1);

  const darken = function (color) {
    return darkMode
      ? d3.rgb(color).brighter(3).toString()
      : d3.rgb(color).darker(2).toString();
  };

  let darkMode = null;

  let deltaRad = 0

  let prevWheeled = ""

  const initTransform = {}

  const props = {
    linkBaseColor: "#aaa",
    linkWidth: 1,
    linkWidthHighlight: 3,
    nodeColor: () => (darkMode ? "#eee" : "#444"),
    nodeFontSize: 10,
    nodeFontSizeBold: 16,
    nodeMargin: 2,
    inputBgColor: "#ccc",
    controlBoxBg: () => (darkMode ? "#666" : "#fff"),
    controlBoxColor: () => (darkMode ? "#fff" : "#111"),
    controlBoxColor2: () => (darkMode ? "#444" : "#eee"),
    inputBgAll: "#444",
    colorHighlight: "red",
    windowHeight: window.innerHeight,
    windowWidth: window.innerWidth,
    arcWidth: 5,
    arcMargin: 0,
    bgColor: () => (darkMode ? "#222" : "#fff"),
    groupLabelSize: 24,
    groupLabelRatio: 0.45,
    groupLinesColor: () => darkMode ? "#fff" : "#111",
    tooltipBg: () => (darkMode ? "#ddd" : "#fff"),
    textEstimateL: 200,
  };


  const nodesNumber = treeData.children
    .map((d) => d.children.length)
    .reduce((sum, x) => sum + x);

  const radius =
    ((nodesNumber + treeData.children.length) * (props.nodeFontSize + props.nodeMargin)) / (2 * Math.PI); // props.windowHeight / 2;

  const controlBoxHeight = d3.select(".control").node().getBoundingClientRect()
    .height;

  const tree = d3.cluster().size([2 * Math.PI, radius]);

  const root = tree(bilink(d3.hierarchy(treeData)));

  const line = d3
    .lineRadial()
    .curve(d3.curveBundle.beta(0.85))
    .radius((d) => d.y - props.arcWidth - props.arcMargin)
    .angle((d) => d.x);




  const svg = d3
    .select("#chart")
    .append("svg")
    .attr("class", ".edgebundling-svg")
    .style("position", "absolute")
    .style("z-index", "-1")
    .style("top", 0)
    .style("left", 0)
    .style("width", "100%")
    .style("height", "100%")
    .style("background-color", props.bgColor);

  svg.on("click", () => {
    enableSoundNotice.style("display", "none")
  })


  const layerBg = svg.append("g").attr("class", "bg");
  const layerChart = svg.append("g")
  const layerEdgeBundling = layerChart.append("g").call(transformInit)
  const layerWheel = layerEdgeBundling.append("g").attr("class", "wheel")
  const layerNodes = layerEdgeBundling.append("g").attr("class", "nodes");
  const layerLinks = layerEdgeBundling.append("g").attr("class", "links");
  const layerGroupLabel = layerEdgeBundling.append("g").attr("class", "arcs");

  const bg = layerBg
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", props.windowWidth)
    .attr("height", props.windowHeight)
    .attr("fill", props.bgColor)
    .on("click", clickBg);

  const wheel = layerWheel.append("circle")
    .attr("cx", 0)
    .attr("cy", 0)
    .attr("r", radius + props.textEstimateL * 2)
    .attr("stroke", "none")
    .attr("fill", props.bgColor)

  const wheelNeedle = layerWheel.append("rect")
    .attr("x", radius)
    .attr("y", -props.nodeFontSize * 1.5)
    .attr("width", props.textEstimateL)
    .attr("height", props.nodeFontSize * 2)
    .attr("fill", props.groupLinesColor)
    .attr("opacity", 0.1)

  const documentCounts = d3.select("#document-counts").html(rawData.length);

  const inputsDimension = d3
    .select("#similarity-dimension")
    .selectAll("div")
    .data(similarityDimensions)
    .join("div")
    .style("background-color", props.inputBgColor)
    .attr("class", "button");

  const inputsDimensionText = inputsDimension
    .append("span")
    .style("background-color", "transparent")
    .attr("class", "noselect")
    .attr("id", (k) => k)
    .html((k) => k)
    .attr("id", (k) => "dimension-" + k)
    .attr("class", (k) => "dimension")
    .attr("value", (k) => k)
    .on("click", handleInput);

  const inputsGroupBy = d3
    .select("#group-by")
    .append("div")
    .attr("class", "button")
    .style("background-color", props.inputBgColor)
    .html(groupBy);

  const node = layerNodes
    .attr("font-family", "sans-serif")
    .attr("font-size", props.nodeFontSize)
    .selectAll("g")
    .data(root.leaves())
    .join("g")
    .attr("class", "node-g")
    .attr(
      "transform",
      (d) => `rotate(${(d.x / Math.PI) * 180 - 90}) translate(${d.y},0)`
    )
    .append("text")
    .attr("class", "node-text")
    .attr("id", d => "node" + `${d.data.id}`)
    .attr("fill", props.nodeColor)
    .style("cursor", "pointer")
    .style("pointer-events", "click")
    .attr("dy", "0.31em")
    .attr("x", (d) => (d.x < Math.PI ? 6 : -6))
    .attr("text-anchor", (d) => (d.x < Math.PI ? "start" : "end"))
    .attr("transform", (d) => (d.x >= Math.PI ? "rotate(180)" : null))
    .text((d) => excerpt(d.data.text))
    .each(function (d) {
      d.text = this;
      d.group = d.data.group;
    })
    .on("click", nodeClicked)
    .on("mouseover", nodeOvered)
    .on("mouseout", nodeOuted);

  const link = layerLinks
    .attr("stroke", props.linkBaseColor)
    .attr("stroke-width", props.linkWidth)
    .attr("fill", "none")
    .selectAll("path")
    .data(root.leaves().flatMap((leaf) => leaf.outgoing))
    .join("path")
    .attr("d", ([i, o]) => {
      return line(i.path(o));
    })
    .each(function (d) {
      this.similarity = d.similarity;
      this.similarity_dimension = d.similarity_dimension;
      d.path = this;
    })
    .attr("opacity", function (d) {
      return this.similarity / 100;
    });

  for (const leaves of root.children) {
    const childrensX = leaves.children.map((c) => c.x);
    leaves.groupName = leaves.data.groupName;
    leaves.arcX = {
      start: d3.min(childrensX),
      end: d3.max(childrensX),
    };
  }

  const groupG = layerGroupLabel
    .selectAll("g")
    .data(root.children.map(d => {
      const obj = d.arcX
      obj.groupName = d.groupName
      return obj
    }))

  const groupLabelLines = groupG
    .join("path")
    .style("cursor", "pointer")
    .attr("id", (d) => ("group-label" + d.groupName).replace(/\s/g, ""))
    .attr("class", "group-label")
    .attr("stroke", props.groupLinesColor)
    .attr("fill", "none")
    .attr("d", (d) => {
      const radian = d.start + (d.end - d.start) / 2
      return drawLabelLines(radian, d.start, d.end, d.groupName)
    })


  function drawLabelArc(g) {
    const radNode = props.nodeFontSize / (radius + props.textEstimateL) / 2
    const arc =
      d3.arc()
        .innerRadius(radius + props.textEstimateL)
        .outerRadius(radius + props.textEstimateL + props.arcWidth)
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

  const groupLabelArc = groupG
    .join("path")
    .style("cursor", "pointer")
    .attr("id", (d) => ("arc" + d.groupName).replace(/\s/g, ""))
    .attr("fill", props.groupLinesColor)
    .call(drawLabelArc)

  const groupLabelBg = groupG
    .join("path")
    .attr("fill", props.bgColor)
    .attr("stroke", "none")
    .attr("d", (d) => {
      const radian = d.start + (d.end - d.start) / 2
      return drawLabelBg(radian, d.start, d.end, d.groupName)
    })

  const groupText = groupG
    .join("text")
    // .style("pointer-events", "none")
    .attr("rotate", (d) =>
      (d.start + (d.end - d.start) / 2) > Math.PI ? "180" : "0")

  const groupTextPath = groupText
    .append("textPath")
    .attr("xlink:href", (d) => ("#group-label" + d.groupName).replace(/\s/g, ""))
    .style("font-size", props.groupLabelSize)
    .style("text-anchor", "end")
    .style("alignment-baseline", (d) => "middle")
    .attr("fill", "white")
    .text((d) => d.start + (d.end - d.start) / 2 > Math.PI
      ? d.groupName.split("").reverse().join("")
      : d.groupName)
    .attr("startOffset", "100%")

  const enableSoundNotice = d3.select(".toggle-sound-notice")
    .style("background-color", props.groupLinesColor)
    .style("color", props.bgColor)

  layerBg.call(
    d3
      .zoom()
      .extent([
        [0, 0],
        [props.windowWidth, props.windowHeight],
      ])
      .scaleExtent([0, 20])
      .on("zoom", zoomed)
  );


  layerChart.call(
    d3
      .zoom()
      .extent([
        [0, 0],
        [props.windowWidth, props.windowHeight],
      ])
      .scaleExtent([0, Infinity])
      .on("zoom", wheeled)
  );

  const audioPromise = document.getElementById("audio-wheel")
  d3.select("#audio-wheel-button").on("click", () => {
    audioPromise.currentTime = 0
    audioPromise.play()
  })


  function wheeled({ transform, sourceEvent }) {
    const nodeRad = props.nodeFontSize / radius
    const rotationY = Math.abs(Math.floor(sourceEvent.wheelDeltaY / 120))


    // let i = 0
    // function rotationLoop() {
    //   setTimeout(function () {
    //     rotateWheel()
    //     i++
    //     if (i < rotationY) {
    //       rotationLoop()
    //     }
    //   }, 200)
    // }

    // rotationLoop()
    rotateWheel()


    function rotateWheel() {
      const e = sourceEvent.type == "mousemove" ? sourceEvent.movementY :
        sourceEvent.type == "wheel" ? sourceEvent.deltaY : 0

      const assumedSegment = ((root.leaves().length + root.children.length)) * 2 + 1

      deltaRad = e > 0 ? deltaRad + 2 * Math.PI / assumedSegment : deltaRad - 2 * Math.PI / assumedSegment

      layerLinks.attr("transform", (d) => `rotate(${((deltaRad) / Math.PI) * 180}) `)

      layerNodes.attr("transform", (d) => `rotate(${((deltaRad) / Math.PI) * 180}) `)

      node
        .attr("x", (d) => (Math.sin(d.x + deltaRad) > Math.sin(Math.PI) ? 6 : -6))
        .attr("text-anchor", (d) => (Math.sin(d.x + deltaRad) > Math.sin(Math.PI) ? "start" : "end"))
        .attr("transform", (d) => (Math.sin(d.x + deltaRad) > Math.sin(Math.PI) ? null : "rotate(180)"))

      groupLabelLines.attr("d", (d, i) => {
        return drawLabelLines(newRad(deltaRad, d), d.start, d.end, d.groupName)
      })

      groupLabelArc.attr("transform", () => `rotate(${deltaRad / (Math.PI * 2) * 360})`)

      groupLabelBg
        .attr("d", (d) => {
          return drawLabelBg(newRad(deltaRad, d), d.start, d.end, d.groupName)
        })

      groupText
        .attr("rotate", (d) => {
          return newRad(deltaRad, d) > Math.PI ? "180" : "0"
        })

      groupTextPath
        .text((d) => {
          return newRad(deltaRad, d) > Math.PI
            ? d.groupName.split("").reverse().join("")
            : d.groupName
        })


      // Fake click and hover event triggering tooltip on wheel
      //
      const nodeFocus = root.leaves()
        .filter(d => {
          let radianFocus = (d.x + deltaRad) % (Math.PI * 2)
          radianFocus = radianFocus < 0 ? 2 * Math.PI + radianFocus : radianFocus
          return radianFocus > Math.PI / 2 - props.nodeFontSize / radius &&
            radianFocus < Math.PI / 2 + props.nodeFontSize / radius
        })[0]

      if (nodeFocus !== undefined) {
        const fakeClickEvent = { pageX: sourceEvent.clientX + 500, pageY: controlBoxHeight + 30 }
        nodeClicked(fakeClickEvent, nodeFocus)

        $(".node-text").d3Mouseout()
        $(`#node${nodeFocus.data.id}`).d3Mouseover()

        if (prevWheeled !== nodeFocus.data.id) {
          $(`#audio-wheel-button`).d3Mouseclick()
          prevWheeled = nodeFocus.data.id
        }

      }


    }

    // Processing rotation value
    function newRad(deltaRad, d) {
      let radian = ((d.start + (d.end - d.start) / 2) + deltaRad) % (Math.PI * 2)

      radian = radian > 0 ? radian : 2 * Math.PI + radian
      return radian
    }
  }

  const tooltip = d3
    .select("body")
    .append("div")
    .attr("id", "tooltip")
    .attr("class", "tooltip")
    .style("position", "absolute")
    .style("z-index", "10")
    .style("visibility", "hidden");

  init();

  function init() {
    dimensions = ["all"];
    dimensionAll = !dimensionAll;
    d3.select(`#dimension-all`).classed("active", dimensionAll ? true : false);
    handleDarkMode();
    inputsUpdate();
    updateLink();
  }

  function handleDarkMode() {
    const toggleDark = d3.select("#toggle-dark");
    const localStorage = window.localStorage;
    darkMode =
      localStorage.darkMode === undefined
        ? toggleDark.node().checked
        : localStorage.darkMode;

    darkMode = darkMode === "true" ? true : false;

    toggleDark.node().checked = darkMode;
    setColorMode();

    toggleDark.on("click", function (e) {
      darkMode = this.checked;
      window.localStorage.darkMode = darkMode;
      setColorMode();
    });
  }

  function setColorMode() {
    bg.attr("fill", props.bgColor);
    node.attr("fill", props.nodeColor);
    tooltip
      .style("background-color", props.tooltipBg)
      .style("color", props.tooltipBg);

    d3.select(".document-counts")
      .style("background-color", "transparent")
      .style("color", props.controlBoxColor);

    d3.selectAll(".inputs-bg").style("background-color", "transparent");
    d3.selectAll(".inputs-bg h4").style("color", props.controlBoxColor);
    d3.selectAll(".group-by")
      .style("background-color", props.controlBoxColor2)
      .style("border", "none");
    d3.selectAll(".similarity-dimension")
      .style("background-color", props.controlBoxColor2)
      .style("border", "none");

    wheel.attr("fill", props.bgColor)

    wheelNeedle.attr("fill", props.groupLinesColor)

    groupLabelBg.attr("fill", props.bgColor)

    groupLabelLines.attr("stroke", props.groupLinesColor)

    groupTextPath.attr("fill", props.groupLinesColor)

    groupLabelArc
      .attr("fill", props.groupLinesColor)

    enableSoundNotice
      .style("background-color", props.groupLinesColor)
      .style("color", props.bgColor)

  }

  // Functionalities
  function handleInput(event, k) {
    d3.selectAll(".dimension").classed("active", false);
    d3.select(`#dimension-${k}`).classed("active", true);

    if (k === "all") {
      dimensionAll = !dimensionAll ? !dimensionAll : dimensionAll;
      dimensions = dimensionAll ? ["all"] : [];
      d3.selectAll(".dimension").classed("active", false);
      d3.select(`#dimension-all`).classed(
        "active",
        dimensionAll ? true : false
      );
    } else {
      if (dimensionAll) {
        dimensionAll = !dimensionAll;
        dimensions = [];
        d3.select(`#dimension-all`).classed("active", false);
      }
      setDimensions(k);
      if (dimensions.length === 0) {
        dimensions = ["all"];
        dimensionAll = !dimensionAll;
        d3.select(`#dimension-all`).classed("active", true);
        d3.select(`#dimension-all`).classed(
          "active",
          dimensionAll ? true : false
        );
      }
    }
    updateLink();
    //setDimensions(k);
    inputsUpdate();
  }

  function inputsUpdate() {
    inputsDimension.style("background-color", (k) => {
      if (k === "all") {
        return dimensionAll ? linkColors(k) : props.inputBgColor;
      } else {
        return dimensions.includes(k) ? linkColors(k) : props.inputBgColor;
      }
    });
  }

  function updateLink() {
    link
      .attr("stroke", function (d) {
        return dimensions.includes(d.similarity_dimension)
          ? linkColors(d.similarity_dimension)
          : props.linkBaseColor;
      })
      .attr("opacity", function (d) {
        return dimensions.includes(d.similarity_dimension)
          ? (d.similarity / 100).toFixed(2)
          : 0;
      });
  }

  function zoomed({ transform }) {
    layerChart
      .attr(
        "transform",
        `translate(${transform.x},
           ${transform.y}) 
                  scale(${transform.k}) `
      );

  }

  function excerpt(text) {

    text = text.split(" ").slice(0, 5).join(" ")
    const textLength = text.length * props.nodeFontSize
    return textLength > 400 ? text.slice(0, Math.floor(400 / props.nodeFontSize)) : text;
  }

  function excerptBold(text) {
    text = text.split(" ").slice(0, 5).join(" ")
    const textLength = text.length * props.nodeFontSizeBold
    return textLength > 400 ? text.slice(0, Math.floor(400 * 0.9 / props.nodeFontSizeBold)) : text;
  }

  function nodeOvered(event, d) {
    link.style("mix-blend-mode", null);
    d3.select(this).attr("font-weight", "bold")
      .attr("font-size", props.nodeFontSizeBold)
      .text(excerptBold(d.data.text))

    let pairedNodes = d.outgoing.concat(d.incoming).filter((k) => {
      return dimensionAll
        ? k.similarity_dimension === "all"
        : k.similarity_dimension !== "all";
    });

    d3.selectAll(
      pairedNodes.map((k) => {
        return k.path;
      })
    )
      .attr("stroke-width", props.linkWidthHighlight)
      .attr("stroke", (k) => {
        return dimensions.includes(k.similarity_dimension)
          ? darken(linkColors(k.similarity_dimension))
          : darken(props.linkBaseColor);
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
        return dimensions.includes(val) ? linkColors(val) : props.nodeColor;
      });
  }

  function nodeOuted(event, d) {
    const pairedNodes = d.outgoing.concat(d.incoming);

    link.style("mix-blend-mode", null);
    d3.select(this).attr("font-weight", null)
      .attr("fill", props.nodeColor)
      .attr("font-size", props.nodeFontSize)
      .text(excerpt(d.data.text));
    d3.selectAll(pairedNodes.map((d) => d.path))
      .attr("stroke", (d) => {
        return dimensions.includes(d.similarity_dimension)
          ? linkColors(d.similarity_dimension)
          : props.linkBaseColor;
      })
      .attr("stroke-width", props.linkWidth);
    d3.selectAll(
      pairedNodes.map(([i, o]) => (i.data.id === d.data.id ? o.text : i.text))
    )
      .attr("fill", props.nodeColor)
      .attr("font-weight", null);
  }

  function nodeClicked(event, d) {
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
                    <div style="background-color:${linkColors(
            val.similarity_dimension
          )};">
                      <p style="color:white;">${val.similarity}%</p>
                    </div>
                    <p><span ><strong>${data.group}</strong></span><br>
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

    tooltipPosition(event);
  }

  function transformInit(g) {
    initTransform.k =
      (props.windowHeight - controlBoxHeight) / 2 / (radius + props.textEstimateL);
    initTransform.x = props.windowWidth / 2
    initTransform.y = (radius + props.textEstimateL * 2) * initTransform.k
    g.attr(
      "transform",
      `translate(${initTransform.x},${initTransform.y
      }) scale(${initTransform.k})`
    );
  }

  function clickBg(event, d) {
    $(".node-text").d3Mouseout()
    d3.select("#tooltip").style("visibility", "hidden");
  }

  function drawLabelLines(deg, start, end, groupName) {
    const outer = radius + props.textEstimateL
    const assumedLeaves = root.leaves().length + root.children.length
    let factor = ((deg / (Math.PI * 2) * assumedLeaves) % (assumedLeaves / 2)) - (assumedLeaves / 4)
    factor = deg >= Math.PI ? factor : -factor

    const y2 = -factor * props.nodeFontSize * 2
    let textLen =
      groupName.length * props.groupLabelSize * props.groupLabelRatio;
    textLen = deg >= Math.PI ? -textLen : textLen

    return d3.line()([
      // [Math.sin(deg)*radius,-Math.cos(deg)*radius],
      [Math.sin(deg) * outer, -Math.cos(deg) * outer],
      [Math.sin(deg) * (outer + 100), y2],
      [Math.sin(deg) * (outer + 100) + textLen * 1.3, y2],
    ])
  }

  function drawLabelBg(deg, start, end, groupName) {
    const outer = radius + props.textEstimateL
    const assumedLeaves = root.leaves().length + root.children.length
    let factor = ((deg / (Math.PI * 2) * assumedLeaves) % (assumedLeaves / 2)) - (assumedLeaves / 4)
    factor = deg > Math.PI ? factor : -factor

    const y2 = -factor * props.nodeFontSize * 2
    let textLen =
      groupName.length * props.groupLabelSize * props.groupLabelRatio;
    textLen = deg > Math.PI ? -textLen : textLen

    const halfFont = props.groupLabelSize / 2

    return d3.line()([
      [Math.sin(deg) * (outer + 100) + textLen * 0.05, y2 - halfFont],
      [Math.sin(deg) * (outer + 100) + textLen * 0.05, y2 + halfFont],
      [Math.sin(deg) * (outer + 100) + textLen * 1.3, y2 + halfFont],
      [Math.sin(deg) * (outer + 100) + textLen * 1.3, y2 - halfFont],
    ])
  }



  function tooltipPosition(event) {
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
