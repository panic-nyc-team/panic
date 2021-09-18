function transformInit() {


    const mainElements = d3.select(".chart")
        .selectChildren().nodes().map(d => `#${d.id}`)
    for (let el of mainElements) {
        d3.select(el).call(setInitTransform)
    }

    function setInitTransform(g) {
        const k =
            (props.windowHeight - controlBoxHeight) / 2 / (radius + props.textEstimateL);
        const x = props.windowWidth / 2
        const y = (radius + props.textEstimateL * 2) * k

        g.attr("transform", `translate(${x},${y}) scale(${k})`);
    }
}


addZoom()

function addZoom() {
    const zoom =
        d3
            .zoom()
            .extent([
                [0, 0],
                [props.windowWidth, props.windowHeight],
            ])
            .scaleExtent([0, 20])
            .on("zoom", zoomed)

    let zoomedElement = layerBg.call(zoom);

    d3.select("#toggle-center-button").on('click', function () {
        zoomedElement.transition()
            .duration(750).call(zoom.transform, d3.zoomIdentity);
    })

    function zoomed({ transform }) {
        layerChart
            .attr(
                "transform",
                `translate(${transform.x},
           ${transform.y}) 
                  scale(${transform.k}) `
            );
    }
}