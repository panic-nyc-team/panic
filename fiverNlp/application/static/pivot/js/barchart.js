function barChart(layerBarChart, json, darkMode) {
  // Chart Settings
  const chart = {
    height: 200,
    width: 500,
    margin: {
      top: 50,
      bottom: 30,
      left: 50,
      right: 50,
    },
    barFill: () => (darkMode ? "#aaa" : "#ccc"),
    lineStroke: () => (darkMode ? "#3978e6" : "#3978e6"),
    bgColor: () => (darkMode ? "#222" : "#fff"),
    textColor: () => (darkMode ? "#fff" : "#111"),
  };

  //Prepare Data
  let data = [];

  json.forEach((d) => {
    const obj = d;
    const date = new Date(d.publish_date);
    obj.date_published = date;
    obj.date_string = date.toDateString();

    data.push(obj);
  });

  let dataRolled = d3.rollup(
    data,
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

  //Scales
  const counts = [...dataRolled.values()].map((d) => d.length);
  const closingPrice = [...dataRolled.values()].map((d) => d.closing_price);

  const x = d3
    .scaleTime()
    .domain([d3.min(dataRolled.keys()), d3.max(dataRolled.keys())])
    .rangeRound([0, chart.width - chart.margin.right - chart.margin.left]);

  const y1 = d3
    .scaleLinear()
    .domain([0, d3.max(counts)])
    .range([chart.height - chart.margin.top - chart.margin.bottom, 0]);

  const y2 = d3
    .scaleLinear()
    .domain([d3.min(closingPrice), d3.max(closingPrice)])
    .range([chart.height - chart.margin.top - chart.margin.bottom, 0]);

  // Bar Width
  const oneDay = 24 * 60 * 60 * 1000;
  const dateDiffMs = d3.max(dataRolled.keys()) - d3.min(dataRolled.keys());
  const dateNumDiff = Math.round(Math.abs(dateDiffMs / oneDay));
  let barWidth =
    (chart.width - chart.margin.left - chart.margin.right) / dateNumDiff;
  const barPadding = 0.2 * barWidth;
  barWidth = barWidth - barPadding;
  //Axis
  //
  const xAxis = d3.axisBottom(x).ticks(6).tickFormat(d3.timeFormat("%b/%d"));
  const y1Axis = d3.axisLeft(y1);

  function formatTickY2Axis(d) {
    return this.parentNode.nextSibling ? `\xa0${d}` : `$${d}`;
  }
  const y2Axis = d3.axisRight(y2).tickFormat(formatTickY2Axis);

  // Draw Chart
  layerBarChart = layerBarChart.append("g").attr("id", "bar-chart");

  layerBarChart.attr(
    "transform",
    `translate(${window.innerWidth * 0.55},${0})`
  );

  layerBarChart
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", chart.width)
    .attr("height", chart.height)
    .attr("fill", chart.bgColor);

  layerBarChart
    .append("g")
    .attr("id", "bars")
    .attr("fill", "steelblue")
    .attr("fill-opacity", 0.8)
    .selectAll("rect")
    .data([...dataRolled])
    .join("rect")
    .attr("fill", chart.barFill)
    .attr("x", (d) => chart.margin.left + x(d[0]) - barWidth / 2)
    .attr("y", (d) => chart.margin.top + y1(d[1].length))
    .attr("width", barWidth)
    .attr("height", (d) => y1(0) - y1(d[1].length));

  const line = d3
    .line()
    .x((d) => {
      return x(d[0]) + chart.margin.left;
    })
    .y((d) => {
      return y2(d[1].closing_price) + chart.margin.top;
    });

  layerBarChart
    .append("path")
    .attr("fill", "none")
    .attr("stroke", chart.lineStroke)
    .attr("stroke-miterlimit", 1)
    .attr("stroke-width", 2)
    .attr("d", line([...dataRolled].sort((a, b) => a[0] - b[0])));

  layerBarChart
    .append("g")
    .attr("id", "xAxis")
    .call(xAxis)
    .attr(
      "transform",
      `translate(${chart.margin.left},${chart.height - chart.margin.bottom})`
    )
    .style("color", chart.textColor);

  layerBarChart
    .append("g")
    .attr("id", "y1Axis")
    .call(y1Axis)
    .attr(
      "transform",
      `translate(${chart.margin.left - barWidth / 2 - barPadding},${
        chart.margin.top
      })`
    )
    .style("color", chart.textColor)
    .call((g) => g.select(".domain").remove());

  layerBarChart
    .append("g")
    .attr("id", "y2Axis")
    .call(y2Axis)
    .attr(
      "transform",
      `translate(${
        chart.width - chart.margin.left + barWidth / 2 + barPadding
      },${chart.margin.top})`
    )
    .style("color", chart.textColor)
    .call((g) => g.select(".domain").remove());

  layerBarChart
    .append("g")
    .attr("id", "barchart-title")
    .append("foreignObject")
    .attr("width", chart.width - chart.margin.left - chart.margin.right)
    .attr("height", chart.margin.top)
    .attr("x", chart.margin.left)
    .attr("y", 0)
    .style("font-size", 16)
    .append("xhtml:div")
    .attr("class", "bar-chart-title")
    .append("span")
    .style("color", chart.textColor)
    .html("Daily Document Volume With Closing Share Price (ADIL)");
}
