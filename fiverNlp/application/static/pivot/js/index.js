// The app works with forceSimulation layout provided by d3.
// It runs like a ticking clock in simulation.on() function
// All the nodes position are controlled inside that function
// Nodes clustering and positioning is set by simulation.force() functions
// There are 4 forces working: charge, collide, position-x, position-y
// The app input and output works with a)updateTree which controls hierarchy
// and updateSide which controls enrichment
// Clusters positioning is in getFoci (hierarchy) and getFociSide (enrichment)
// Tooltip position is set in tooltip() function and the content is in handleNodeClick.
// Set what to show there.

d3.json(query_name)
  .then(function (json) {
    //Clean nulls
    const keys = Object.keys(json[0]);
    json.forEach((node, i) => {
      keys.forEach((k) => {
        if (node[k] === null) {
          json[i][k] = "null";
        } else if (node[k] === undefined) {
          json[i][k] = "null";
        }

        json[i]["type"] = "main";
      });
    });
    DrawChart(json);
  })
  .catch(function (error) {
    console.log(error);
  });

function DrawChart(dataDocuments) {
  // Drawing the pivot chart
  // Input are controlled by this.groupBy(hierarchy) and this.extras (enrichment)
  let groupBy = ["site_type", "site"]; // Set default hiearchy attribute
  this.extras = [];
  let data = dataDocuments;

  //this.keys = ["entity", "author", "sentiment", "site", "site_type", "topic"]; //Object.keys(data[0]);
  this.keys = Object.keys(data[0]).filter((d) => d !== "type"); // Change this to set which attribute to show in enrichment bart. Example above.

  this.addGroupBy = (value) => {
    // Function to interact with jquery in drag and drop input function
    groupBy.push(value);
    clearColoring();
    updateTree();
    updateSide();
  };
  this.removeGroupBy = (value) => {
    // Function to interact with jquery in drag and drop input function
    groupBy = groupBy.filter((d) => d !== value);
    clearColoring();
    updateTree();
    updateSide();
  };

  this.setExtras = (k) => {
    // Idem but with enrichment
    if (this.extras.includes(k)) {
      this.extras = this.extras.filter((v) => v !== k);
    } else {
      this.extras.push(k);
    }
    clearColoring();
    updateSide();
  };

  //PREPARE Inputs
  manageInputs(groupBy); // Initiate the drag and drop inputs in ./js/dragdropinput.js

  //Static Icons
  // Set icons accordingly
  const pathIcon = "/static/pivot/static/icons/";

  const iconUrl = {
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

  let darkMode = null;

  // Settings for hierarchy nodes
  const props = {
    nodeFill: () => (darkMode ? "#333" : "#fff"),
    imageFilter: () => (darkMode ? "invert(1)" : "invert(0)"),
    nodeRadius: 10,
    nodeStroke: () => (darkMode ? "#777" : "#aaa"),
    nodeStrokeWidth: 2,
    bgColor: () => (darkMode ? "#222" : "#fff"),
    textColor: () => (darkMode ? "#fff" : "#111"),
    imageNodeRatio: 1.3,
    hullFill: () => (darkMode ? "#222" : "#fff"),
    inputBg: () => (darkMode ? "#ddd" : "#fff"),
    hullStroke: "#aaa",
    hullStrokeWidth: 1,
    hullOpacity: 0.8,
    sideNodeRadius: 20,
    sideFontColor: () => (darkMode ? "#fff" : "#222"),
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

  //Draw SVG
  fociSide = {}; // Position of enrichment clusters

  let clusterMap = null;

  const width = window.innerWidth;
  const height = window.innerHeight;

  //
  let hierarchyCenter = [width / 2, height / 2];

  let clickedNode = {}; // To remember which node just clicked

  let nodesExtras = []; // Nodes of enrichment

  const brighten = function (color) {
    return d3.rgb(color).brighter(0).toString();
  };

  const svg = d3
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

  //Layers
  const layerMainBg = svg.append("g").attr("id", "layerMainBg");
  const layerMain = svg.append("g").attr("id", "layerMain");
  const layerTree = svg.append("g").attr("id", "layerTree");

  //Zoom control on the svg
  svg.call(
    d3
      .zoom()
      .extent([
        [0, 0],
        [width, height],
      ])
      .scaleExtent([0, 20])
      .on("zoom", zoomed)
  );

  //layerMain
  // Set layer for background
  let bg = layerMainBg
    .append("g")
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", width)
    .attr("height", height)
    .attr("fill", props.bgColor)
    .on("click", clearColoring);

  //Simulation
  let nodes = data.map((d, index) => {
    const obj = d;
    obj.id = index;
    return obj;
  });

  let links = [];

  const charge = (strength, distance) => {
    // Force for nodes to attract each other
    return d3.forceManyBody().strength(strength).distanceMax(distance);
  };
  const collide = (collisionVal) => {
    // Force for nodes to repell each other
    return d3.forceCollide().radius(collisionVal);
  };
  const posX = (fX, strength) => {
    // Force for cluster
    // Usage:
    // Position of cluster center. Nodes will gravitate toward fX.
    // Strength of the force is weaker approaching 0, and stronger towards 1. For balanced force, we use 0.1.
    return d3.forceX(fX).strength(strength);
  };
  const posY = (fY, strength) => {
    // Idem but in Y.
    return d3.forceY(fY).strength(strength);
  };

  // isolateForce is to set force applying just to certain nodes. In this case, the hierarchy nodes
  // force link is set to 0 so the links do nothing to nodes.
  const simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .strength(0)
    )
    .force(
      "charge",
      isolateForce(
        charge(-props.nodeRadius * 1.5, props.nodeRadius * 50),
        "main"
      )
    )
    .force("collide", isolateForce(collide(props.nodeRadius * 1.1), "main"));

  // Simulation tree

  let treeColors = d3.scaleOrdinal().range(d3.schemeCategory10);
  let treeLinks = [];
  let treeNodes = [];

  const treeSimulation = d3.forceSimulation(treeNodes);

  //Links

  let link = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "links")
    .selectAll("line");

  let treeLink = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "tree-line")
    .selectAll("line");
  //Hull or cell wrapping the node clusters in hierarchy

  const mainhullG = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "main-hull")
    .attr("class", "hulls");

  //Side
  // Hull for clusters in enrichment
  let sidehullG = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "side-hull")
    .attr("class", "side-hulls");

  let mainHulls = mainhullG.selectAll("path");

  let sideHulls = sidehullG.selectAll("path");

  // Side Hulls Text
  let sideHullsText = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "side-hull-text")
    .selectAll("foreignObject");

  //Side Node Text
  let sideNodeText = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "side-node-text")
    .selectAll("foreignObject");

  //Tooltip
  const tooltip = d3
    .select("body")
    .append("div")
    .attr("id", "tooltip")
    .attr("class", "tooltip")
    .style("position", "absolute")
    .style("z-index", "10")
    .style("visibility", "hidden");

  //
  let treeNode = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "tree-node")
    .selectAll("circle");
  let treeLabel = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "tree-label-text")
    .selectAll("foreignObject");
  //

  //Main Nodes
  let node = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "nodes")
    .attr("class", "node")
    .attr("stroke", props.nodeStroke)
    .attr("fill", props.nodeFill)
    .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("id", "mainNodes")
    .attr("r", props.nodeRadius)
    .on("click", (event, d) => {
      if (d.type === "main") {
        handleNodeClick(event, d);
        node.attr("stroke", (node) =>
          node.id === d.id ? props.linestrokeHighlight : props.nodeStroke()
        );
      }
    });

  let nodeImage = layerMain
    .append("g")
    .call(initTransform)
    .attr("id", "node-image")
    .selectAll("image");

  // Start simulation
  // All nodes movement are controlled here
  // as time goes by
  // treeSimulation.alphaDecay(0);

  updateTree();

  treeSimulation.on("tick", treeTick);

  simulation.on("tick", mainTick);
  //Alpha decay to 0 so simulation goes on and on
  simulation.alphaDecay(0.005);

  handleDarkMode();

  // Functionalities

  function initTransform(g) {
    const k = 0.5 / groupBy.length;
    g.attr(
      "transform",
      `translate(${height / 2},${300 + height * k}) scale(${k})`
    );
  }

  function mainTick(e) {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

    const nodeImageShift = (props.nodeRadius * props.imageNodeRatio) / 2;
    const sideImageShift = (props.sideNodeRadius * props.imageNodeRatio) / 2;

    nodeImage
      .attr("height", (d) =>
        d.type === "main"
          ? props.nodeRadius * props.imageNodeRatio
          : props.sideNodeRadius * props.imageNodeRatio
      )
      .attr("x", (d) =>
        d.type === "main" ? d.x - nodeImageShift : d.x - sideImageShift
      )
      .attr("y", (d) =>
        d.type === "main" ? d.y - nodeImageShift : d.y - sideImageShift
      );

    mainHulls.attr("d", (d) => hullPath(d, "main"));

    sideHulls.attr("d", (d) => hullPath(d, "extra"));

    sideNodeText
      .attr("x", (d) => {
        if (d.type === "extra") {
          return d.x - props.sideNodeRadius * 2;
        }
      })
      .attr("y", (d) => {
        if (d.type === "extra") {
          return d.y - props.sideNodeRadius * 4.1;
        }
      });

    sideHullsText
      .attr("width", (d) => fociSide[d.cluster].clusterR)
      .attr("x", (d) => {
        return fociSide[d.cluster].x - fociSide[d.cluster].clusterR / 2;
      })
      .attr(
        "y",
        (d) =>
          fociSide[d.cluster].y -
          fociSide[d.cluster].clusterR -
          props.sideNodeRadius * 2
      );
  }

  function treeTick(e) {
    simulation
      .force(
        "positiion-x",
        isolateForce(
          posX((d) => getFociTree(groupBy, d).x).strength(0.1),
          "main"
        )
      )
      .force(
        "positiion-y",
        isolateForce(
          posY((d) => getFociTree(groupBy, d).y).strength(0.1),
          "main"
        )
      );

    treeLink
      .attr("x1", (d) => (d.source.type === "leaf" ? d.source.cx : d.source.x))
      .attr("y1", (d) => (d.source.type === "leaf" ? d.source.cy : d.source.y))
      .attr("x2", (d) => (d.target.type === "leaf" ? d.target.cx : d.target.x))
      .attr("y2", (d) => (d.target.type === "leaf" ? d.target.cy : d.target.y));

    treeNode
      .attr("cx", (d) => (d.type === "leaf" ? d.cx : d.x))
      .attr("cy", (d) => (d.type === "leaf" ? d.cy : d.y));

    treeLabel
      .attr("x", (d) => d.x - baseTriangle(d.r))
      .attr("y", (d) => d.y - baseTriangle(d.r));
  }

  function getFociTree(groupBy, node) {
    return clusterMap.get(groupBy.map((k) => node[k]).join("-") + "-leaf");
  }

  //Update Chart
  //
  function getTreeData() {
    const hierarchy = groupBy;

    let combinations = hierarchy
      .map((g) => {
        return Array.from(new Set(data.map((node) => node[g])));
      })
      .reduce((a, b) =>
        a.reduce((r, v) => r.concat(b.map((w) => [].concat(v, w))), [])
      );

    combinations =
      hierarchy.length === 1 ? combinations.map((d) => [d]) : combinations;

    let labelCluster = combinations.map((combination) => {
      return {
        grouping: hierarchy,
        combination: combination,
        nodes: data.filter((item) => {
          for (let i = 0; i < hierarchy.length; i++) {
            if (item[hierarchy[i]] !== combination[i]) {
              return false;
            }
          }
          return true;
        }),
      };
    });

    labelCluster = labelCluster.filter((d) => d.nodes.length > 0);

    let treeLinks = Array.from(
      new Set(labelCluster.map((c) => c.combination[0]))
    ).map((d) => {
      return {
        source: "fakeRoot",
        target: d,
        distance: props.sideNodeRadius * 5,
      };
    });

    const treeLabelScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([props.labelRadius, 10 * props.labelRadius]);

    let treeNodes = Array.from(
      labelCluster
        .map((c) => {
          let arr = [];
          for (let i = 0; i < c.combination.length; i++) {
            let combo = c.combination.slice(0, i + 1);
            let grouping = c.grouping.slice(0, combo.length);

            const subNodes = data.filter((item) => {
              for (let i = 0; i < grouping.length; i++) {
                if (item[grouping[i]] !== combo[i]) {
                  return false;
                }
              }
              return true;
            });

            arr.push({
              id: combo.join("-"),
              name: combo[i],
              grouping: combo,
              level: i + 1,
              group: hierarchy[i],
              type: "label",
              r: treeLabelScale(subNodes.length / data.length),
            });
            if (i === c.combination.length - 1)
              arr.push({
                id: combo.join("-") + "-leaf",
                name: combo.join("-") + "-leaf",
                grouping: combo.concat("leaf"),
                level: i + 1,
                group: hierarchy[i],
                type: "leaf",
                clusterSize: subNodes.length,
                r:
                  Math.ceil(Math.sqrt(subNodes.length)) *
                  props.nodeRadius *
                  1.5,
                nodes: subNodes,
              });
          }
          for (let j = 0; j < arr.length - 1; j++) {
            treeLinks.push({
              source: arr[j].id,
              target: arr[j + 1].id,
              type: "tree",
              distance: arr[j].r + arr[j + 1].r,
            });
          }
          return arr;
        })
        .flat()
    ).concat({
      id: "fakeRoot",
      name: "fakeRoot",
      grouping: ["fakeRoot"],
      level: 0,
      group: "fakeRoot",
      type: "root",
      r: props.treeRootRadius,
    });

    let uniqueTreeNodes = [];
    treeNodes.forEach((k) => {
      if (!uniqueTreeNodes.map((d) => d.id).includes(k.id))
        uniqueTreeNodes.push(k);
    });

    return [uniqueTreeNodes, treeLinks];
  }

  function updateTree() {
    treeSimulation
      .force(
        "treeLink",
        d3
          .forceLink(treeLinks)
          .id((d) => d.id)
          .distance((d) => d.distance)
          .strength(1)
      )
      .force(
        "tree-charge",
        d3.forceManyBody().strength((d) => -90 * d.r)
      )
      .force("x", d3.forceX(height / 2))
      .force("y", d3.forceY(height / 2));

    let [newtreeNodes, newtreeLinks] = getTreeData();

    treeNodes = newtreeNodes;

    treeLinks = newtreeLinks;

    treeNode = treeNode
      .data(treeNodes, (d) => d.id)
      .join("circle")
      .attr("r", (d) => d.r)
      .attr("class", (d) => d.name)
      .attr("fill", (d) => brighten(treeColors(d.group)))
      .attr("stroke", props.labelCircleStroke)
      .attr("stroke-width", props.labelStrokeWidth)
      .attr("opacity", (d) => (d.type === "leaf" ? 0 : 1))
      .style("pointer-events", "none");

    function processTreeLinks() {
      const map = new Map(treeNodes.map((d) => [d.id, d]));
      return treeLinks.map((l) => {
        return {
          source: map.get(l.source),
          target: map.get(l.target),
          type: l.type,
          distance: l.distance,
        };
      });
    }

    treeLinks = processTreeLinks();

    treeLink = treeLink
      .data(treeLinks, (l) => [l.source, l.target])
      .join("line")
      .attr("id", "link")
      .attr("stroke", (l) => treeColors(l.target.group))
      .attr("stroke-width", (l) => (2 * (groupBy.length + 1)) / l.target.level)
      .attr("opacity", props.lineopacity);

    treeLabel = treeLabel
      .data(treeNodes.filter((d) => d.type === "label"))
      .join("foreignObject")
      .attr("id", (d, i) => "treelabel-" + d.id)
      .style("pointer-events", "none")
      .attr("width", (d) => baseTriangle(d.r) * 2)
      .attr("height", (d) => baseTriangle(d.r) * 2)
      .style("font-size", (d) => {
        const multiplier = Math.floor(d.name.length / 18) + 1;
        return `${d.r / (2.5 * multiplier)}px`;
      });

    d3.selectAll(".mainlabeldiv").remove();

    const treeLabelSpan = treeLabel
      .append("xhtml:div")
      .attr("class", "mainlabeldiv")
      .append("span")
      .style("color", "white")
      .html((d) => {
        if (d.name !== undefined) {
          return d.name.length > 0 ? d.name : "undefined";
        }
      });

    clusterMap = new Map(
      treeNodes.filter((d) => d.type === "leaf").map((d) => [d.id, d])
    );

    nodeImage = nodeImage
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
      .attr("filter", props.imageFilter);

    mainHulls = mainHulls
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
      .attr("d", (d) => {
        return hullPath(d, "main");
      })
      .attr("fill", props.hullFill)
      .attr("opacity", props.hullOpacity)
      .attr("stroke", props.hullStroke)
      .attr("stroke-width", props.hullStrokeWidth);

    documentCounts();

    const inputsGroups = d3.selectAll(".as-group");
    inputsGroups.each(function () {
      const thisGroup = this.getAttribute("value");
      d3.select(this).style("background-color", treeColors(thisGroup));
    });

    treeSimulation.force("treeLink").links(treeLinks);

    treeSimulation.nodes(treeNodes);

    // treeSimulation.alphaDecay(0.005).velocityDecay(0.6);
    treeSimulation.alpha(1).restart();
  }

  //
  //Update Side bar

  function updateSide() {
    nodes = simulation.nodes();

    const oldExtras = Array.from(
      new Set(nodes.filter((d) => d.type === "extra").map((d) => d.extra))
    );

    const removedExtra = oldExtras.filter((e) => !this.extras.includes(e));

    if (removedExtra[0] !== undefined) {
      nodes = nodes.filter((d) => d.extra !== removedExtra[0]);
      nodesExtras = nodesExtras.filter((d) => d.extra !== removedExtra[0]);
    }

    const newExtras = this.extras.filter((e) => !oldExtras.includes(e));

    fociSide = getFociSide(this.extras);

    newExtras.forEach((extra, i) => {
      obj = {};
      getDimensions(extra).forEach((dimension, j) => {
        if (!nodesExtras.map((d) => d.id).includes(dimension)) {
          obj = {
            id: dimension,
            name: dimension,
            extra: extra,
            type: "extra",
          };
          nodesExtras.push(obj);
        }
      });
    });

    links = [];

    nodesExtras
      .map((nodeSource) => {
        return nodes
          .filter(
            (nodeTarget) => nodeTarget[nodeSource.extra] === nodeSource.id
          )
          .map((nodeTarget) => {
            return {
              source: nodeTarget[nodeSource.extra],
              target: nodeTarget.id,
              type: "side",
            };
          });
      })
      .forEach((arr) => {
        links = links.concat(arr);
      });

    nodes = simulation.nodes().map((node, index) => {
      obj = node;
      if (node.type === "main") {
        obj.id = index;
      }
      return obj;
    });

    nodes = nodes.filter((node) => node.type === "main").concat(nodesExtras);

    node = node
      .data(nodes, (d) => d.id)
      .join("circle")
      .on("click", updateColoring)
      .attr("fill", props.nodeFill)
      .attr("r", (d) =>
        d.type === "main" ? props.nodeRadius : props.sideNodeRadius
      );

    link = link.data(links, (l) => [l.source, l.target]).join("line");

    link
      .attr("id", "link")
      .attr("stroke", props.linestroke)
      .attr("stroke-width", props.linestrokeWidth)
      .attr("opacity", props.lineopacity);

    simulation.nodes(nodes);

    simulation.force("link").links(links);

    simulation
      .force(
        "chargeExtra",
        isolateForce(
          charge(-props.sideNodeRadius * 4.5, props.sideNodeRadius * 4.5),
          "extra"
        )
      )
      .force(
        "collideExtra",
        isolateForce(collide(props.sideNodeRadius * 3.6), "extra")
      )
      .force(
        "positionxExtra",
        isolateForce(
          posX(
            (d) => fociSide[d.extra].x,
            (d) => 0.1 / fociSide[d.extra].forceFactor
          ),
          "extra"
        )
      )
      .force(
        "positionyExtra",
        isolateForce(
          posY(
            (d) => fociSide[d.extra].y,
            (d) => 0.1 / fociSide[d.extra].forceFactor
          ),
          "extra"
        )
      );

    simulation.alpha(1).restart();

    nodeImage = nodeImage
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
      .attr("filter", props.imageFilter);

    let extrasClusters = this.extras.map((g) => {
      const obj = {
        cluster: g,
        nodes: node.filter((d) => d.extra === g),
      };
      obj.counts = obj.nodes.nodes().length;
      return obj;
    });

    sideHulls = sideHulls
      .data(extrasClusters, (d) => d.cluster)
      .join("path")
      .style("cursor", "pointer")
      .attr("id", (d) => "sidehull-" + d.cluster)
      .attr("d", (d) => hullPath(d, "extra"))
      .attr("fill", props.hullFill)
      .attr("stroke", props.hullStroke)
      .attr("stroke-width", props.hullStrokeWidth)
      .attr("opacity", props.hullOpacity)
      .call(
        d3
          .drag()
          .on("start", (e, d) => {})
          .on("drag", (e, d) => {
            simulation
              .force(
                "positionxExtra",
                isolateForce(
                  posX(
                    (node) =>
                      node.extra === d.cluster ? e.x : fociSide[node.extra].x,
                    0.1
                  ),
                  "extra"
                )
              )
              .force(
                "positionyExtra",
                isolateForce(
                  posY(
                    (node) =>
                      node.extra === d.cluster ? e.y : fociSide[node.extra].y,
                    0.1
                  ),
                  "extra"
                )
              );
          })
          .on("end", (e, d) => {
            fociSide[d.cluster].x = e.x;
            fociSide[d.cluster].y = e.y;
          })
      );

    //Side hull text

    sideHullsText = sideHullsText
      .data(extrasClusters)
      .join("foreignObject")
      .attr("width", (d) => fociSide[d.cluster].clusterR)
      .attr("height", props.sideNodeRadius * 2)
      .style("font-size", 20);

    d3.selectAll(".side-hull-text-div").remove();

    const sideHullsTextSpan = sideHullsText
      .append("xhtml:div")
      .attr("class", "side-hull-text-div")
      .append("span")
      .style("color", props.sideFontColor)
      .html((d) => d.counts + " " + d.cluster);
    //

    sideNodeText = sideNodeText
      .data(nodes)
      .join("foreignObject")
      .attr("id", (d, i) => "sideNodeText" + d.extra + i)
      .style("pointer-events", "none")
      .attr("width", props.sideNodeRadius * 4)
      .attr("height", props.sideNodeRadius * 3)
      .style("font-size", 12);

    d3.selectAll(".sideNodeTextDiv").remove();

    const sideNodeTextSpan = sideNodeText
      .append("xhtml:div")
      .attr("class", "side-node-text-div")
      .append("span")
      .style("color", props.sideFontColor)
      .html((d) => d.name);
  }

  function getFociSide(extras) {
    // Set the focal position of each cluster
    const mainNodesOuterRing = d3.max(
      nodes
        .filter((d) => d.type === "main")
        .map((d) =>
          distance(d.x - hierarchyCenter[0], d.y - hierarchyCenter[1])
        )
    );

    const newFociSide = {};
    let prevY = 0;

    extras.forEach((extra, i) => {
      const dimensionNum = getDimensions(extra).length;
      const clusterRadius =
        Math.ceil(Math.sqrt(dimensionNum)) * 2.5 * props.sideNodeRadius;
      let forceFactor = Math.log10(dimensionNum);
      forceFactor = forceFactor > 2 ? forceFactor + 1.5 : forceFactor;
      if (!Object.keys(fociSide).includes(extra)) {
        obj = {
          x: width * 0.7 * 0.5 + mainNodesOuterRing + clusterRadius,
          y: prevY + clusterRadius,
          forceFactor: forceFactor,
        };
      } else {
        obj = {
          x: fociSide[extra].x,
          y: fociSide[extra].y,
          forceFactor: forceFactor,
        };
      }
      prevY = obj.y + clusterRadius + 2 * props.sideNodeRadius;
      newFociSide[extra] = obj;
    });

    return newFociSide;
  }

  function hullPath(data, type) {
    let nodesPos = [];
    const nodeRadius =
      type === "main" ? props.nodeRadius : props.sideNodeRadius;
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

    let cx =
      nodesPos.map((node) => node.x).reduce((sum, x) => sum + x) /
      nodesPos.length;

    let cy =
      nodesPos.map((node) => node.y).reduce((sum, y) => sum + y) /
      nodesPos.length;
    cy = type === "main" ? cy : cy - props.sideNodeRadius;

    const maxR = d3.max(
      nodesPos.map((node) => distance(node.x - cx, node.y - cy))
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
      fociSide[data.cluster].clusterR = r;
    }

    return p;
  }

  function distance(xLength, yLength) {
    return Math.sqrt(xLength * xLength + yLength * yLength);
  }

  function baseTriangle(radius) {
    return Math.cos(Math.PI / 4) * radius;
  }

  function getDimensions(groupName) {
    return Array.from(new Set(data.map((node) => node[groupName])));
  }

  const updateColoring = (event, d) => {
    if (clickedNode === [d.id]) {
      node.attr("fill", props.nodeFill()).attr("stroke", props.nodeStroke());
    } else if (d.type === "main") {
      handleNodeClick(event, d);
      node.attr("stroke", (node) =>
        node.id === d.id ? props.linestrokeHighlight : props.nodeStroke()
      );
    } else if (d.type === "extra") {
      const targetedNodesIds = links
        .filter((l) => l.source.id === d.id)
        .map((l) => l.target.id);
      link.attr("stroke", (l) => {
        let color = props.linestroke;
        if (l.source.id === d.id && l.source.id !== clickedNode.id) {
          color = props.linestrokeHighlight;
        }
        return color;
      });

      node
        .attr("stroke", (c) => {
          let color = props.nodeStroke();
          if (c.id === d.id) {
            if (d.id !== clickedNode.id) {
              color = props.linestrokeHighlight;
            }
          }
          return color;
        })
        .attr("fill", (c) => {
          if (d.id !== clickedNode.id) {
            if (c.id !== d.id) {
              if (targetedNodesIds.includes(c.id) && c.type === "main") {
                return props.linestrokeHighlight;
              }
            }
          }
          return props.nodeFill();
        });

      clickedNode.id = clickedNode.id === d.id ? "" : d.id;
      clickedNode.type = d.type;
    }
  };

  //tooltip function
  //
  function handleNodeClick(event, d) {
    const dateString = new Date(d.publish_date).toDateString();
    d3.select("#tooltip").style("visibility", "visible").html(`
      <ul>
        <li class="tooltip-title">${d.title}</li>
        <li class="tooltip-date">${dateString}</li>
        <li class="tooltip-author">By: ${d.author}</li>
        <li class="tooltip-url"><a href="${d.url}" target="_blank">Source</a></li>
      </ul>
      `);
    tooltipPosition(event);
  }
  // Document Count
  function documentCounts() {
    //Interact with document counts in the inputs
    d3.select("#document-counts text").remove();
    d3.select("#document-counts")
      .append("text")
      .text(() => {
        const groupingSizes = groupBy.map((g) => [
          g,
          new Set(data.map((d) => d[g])).size,
        ]);
        const groupingText = groupingSizes
          .map(([g, size]) => `${size} ${g}`)
          .join(", ");
        return groupingText;
      });
  }

  function tooltipPosition(event) {
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

  function handleDarkMode() {
    const toggleDark = d3.select("#toggle-dark");
    const localStorage = window.localStorage;

    darkMode =
      localStorage.pivotChartDarkMode === undefined
        ? toggleDark.node().checked
        : localStorage.pivotChartDarkMode;

    darkMode = darkMode === "true" ? true : false;

    toggleDark.node().checked = darkMode;
    setColorMode();

    toggleDark.on("click", function (e) {
      darkMode = this.checked;
      window.localStorage.pivotChartDarkMode = darkMode;
      setColorMode();
    });
  }

  function setColorMode() {
    bg.attr("fill", props.bgColor);
    mainHulls.attr("fill", props.hullFill);
    sideHulls.attr("fill", props.hullFill);
    node.attr("fill", props.nodeFill).attr("stroke", props.nodeStroke);
    nodeImage.attr("filter", props.imageFilter);
    d3.select("#document-counts").style("color", props.textColor);
    d3.selectAll(".input-title").style("color", props.textColor);
    barChart(svg, data, darkMode);
    d3.select(".group-by").style("background-color", props.inputBg);
  }

  function zoomed({ transform }) {
    transformScale = transform.k;
    transformX = transform.x;
    transformY = transform.y;

    layerMain.attr(
      "transform",
      `translate(${transform.x},${transform.y}) scale(${transform.k})`
    );
    layerTree.attr(
      "transform",
      `translate(${transform.x},${transform.y}) scale(${transform.k})`
    );
  }

  //
  function clearColoring() {
    d3.select("#tooltip").style("visibility", "hidden");
    node.attr("stroke", props.nodeStroke);
    node.attr("fill", props.nodeFill);
    link.attr("stroke", props.linestroke);
    clickedNode = {};
  }

  //Isolate Force To Certain Node
  function isolateForce(force, nodetype) {
    let initialize = force.initialize;
    force.initialize = function () {
      initialize.call(
        force,
        nodes.filter((node) => node.type === nodetype)
      );
    };
    return force;
  }
}
