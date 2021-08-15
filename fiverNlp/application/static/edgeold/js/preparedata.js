function prepareData(data) {
  let rawData = data;
  let nonUniqueNodes = [].concat.apply(
    [],
    rawData.map((l) => [l.s1, l.s2])
  );

  nonUniqueNodes = nonUniqueNodes.map((d, i) => {
    const group = d.parent_site === undefined ? d.parent_title : d.parent_site;
    const group_url = d.url === undefined ? d.parent_url : d.url;
    return {
      text: d.text,
      group: group,
      group_url: group_url,
      targets: [],
    };
  });

  const nodes = new Map();

  let counterUnique = 0;
  for (let i = 0; i < nonUniqueNodes.length; i++) {
    const obj = nonUniqueNodes[i];
    if (!nodes.has(obj.group + obj.text)) {
      obj.id = counterUnique;
      nodes.set(obj.group + obj.text, obj);
      counterUnique++;
    }
  }

  const uniqueNodes = Array.from(nodes.values());

  const groups = Array.from(new Set(uniqueNodes.map((d) => d.group)));

  let tree = [];

  groups.forEach((g, i) => {
    const obj = {};
    obj.id = i;
    obj.groupName = g;
    obj.children = uniqueNodes.filter((d) => d.group === g);
    tree.push(obj);
  });

  let similarityDimensions = [];

  for (let i = 0; i < rawData.length; i++) {
    const link = rawData[i];
    const s1Group =
      link.s1.parent_site === undefined
        ? link.s1.parent_title
        : link.s1.parent_site;
    const s2Group =
      link.s2.parent_site === undefined
        ? link.s2.parent_title
        : link.s2.parent_site;

    const source = nodes.get(s1Group + link.s1.text);
    const target = nodes.get(s2Group + link.s2.text);

    // if (target.id === 69) {
    //   console.log(i);
    // }

    nodes.get(source.group + source.text).targets.push({
      id: target.id,
      similarity_dimension: link.similarity_dimension,
      similarity: link.similarity,
    });

    if (!similarityDimensions.includes(link.similarity_dimension)) {
      similarityDimensions.push(link.similarity_dimension);
    }
  }

  const finish = new Date();

  return {
    tree: { children: tree },
    similarityDimensions: similarityDimensions,
  };
}
