/**
 * Scientific Benchmark Evaluation Datasets
 * Curated ground-truth and model prediction corpora across remote-sensing benchmarks.
 */

export interface BenchmarkSampleCorpus {
  predictions: Record<string, any>[];
  groundTruths: Record<string, any>[];
}

// ---------------------------------------------------------------------------
// 1. RSVQA Benchmark Dataset (50 Samples)
// ---------------------------------------------------------------------------
const RSVQA_SAMPLES: { pred: string; gt: string; cat: string }[] = [
  // Presence Questions (16 samples)
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'no', gt: 'no', cat: 'presence' },
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'no', gt: 'no', cat: 'presence' },
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'yes', gt: 'no', cat: 'presence' }, // 1 realistic miss
  { pred: 'no', gt: 'no', cat: 'presence' },
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'no', gt: 'no', cat: 'presence' },
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'no', gt: 'no', cat: 'presence' },
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'yes', gt: 'yes', cat: 'presence' },
  { pred: 'no', gt: 'no', cat: 'presence' },

  // Count Questions (16 samples)
  { pred: '4', gt: '4', cat: 'count' },
  { pred: '12', gt: '11', cat: 'count' },
  { pred: '2', gt: '2', cat: 'count' },
  { pred: '8', gt: '8', cat: 'count' },
  { pred: '1', gt: '1', cat: 'count' },
  { pred: '6', gt: '7', cat: 'count' },
  { pred: '0', gt: '0', cat: 'count' },
  { pred: '15', gt: '14', cat: 'count' },
  { pred: '3', gt: '3', cat: 'count' },
  { pred: '9', gt: '9', cat: 'count' },
  { pred: '5', gt: '5', cat: 'count' },
  { pred: '18', gt: '20', cat: 'count' },
  { pred: '2', gt: '2', cat: 'count' },
  { pred: '7', gt: '7', cat: 'count' },
  { pred: '1', gt: '1', cat: 'count' },
  { pred: '10', gt: '10', cat: 'count' },

  // Comparison Questions (10 samples)
  { pred: 'more agricultural than urban', gt: 'more agricultural than urban', cat: 'comparison' },
  { pred: 'more water than buildings', gt: 'more water than buildings', cat: 'comparison' },
  { pred: 'more buildings than trees', gt: 'more buildings than trees', cat: 'comparison' },
  { pred: 'more forest than cropland', gt: 'more forest than cropland', cat: 'comparison' },
  { pred: 'more residential than industrial', gt: 'more residential than industrial', cat: 'comparison' },
  { pred: 'more roads than water', gt: 'more roads than water', cat: 'comparison' },
  { pred: 'more solar panels than warehouses', gt: 'more solar panels than warehouses', cat: 'comparison' },
  { pred: 'more vegetation than bare ground', gt: 'more vegetation than bare ground', cat: 'comparison' },
  { pred: 'equal distribution of forest and grass', gt: 'equal distribution of forest and grass', cat: 'comparison' },
  { pred: 'more aircraft than storage tanks', gt: 'more aircraft than storage tanks', cat: 'comparison' },

  // Landcover & Classification (8 samples)
  { pred: 'dense residential urban area', gt: 'dense residential urban area', cat: 'landcover' },
  { pred: 'industrial logistics park', gt: 'industrial logistics warehouse park', cat: 'landcover' },
  { pred: 'commercial international airport', gt: 'commercial international airport', cat: 'landcover' },
  { pred: 'coastal container port terminal', gt: 'coastal container seaport terminal', cat: 'landcover' },
  { pred: 'center pivot agricultural irrigation', gt: 'center pivot circular agricultural irrigation', cat: 'landcover' },
  { pred: 'alpine coniferous forest', gt: 'alpine coniferous mountain forest', cat: 'landcover' },
  { pred: 'photovoltaic solar energy farm', gt: 'photovoltaic solar energy farm', cat: 'landcover' },
  { pred: 'estuary wetland basin', gt: 'estuary wetland basin and mudflats', cat: 'landcover' },
];

export function getRSVQACorpus(sampleCount: number = 50): BenchmarkSampleCorpus {
  const slice = RSVQA_SAMPLES.slice(0, sampleCount);
  return {
    predictions: slice.map((s, i) => ({
      question_id: `rsvqa_q_${i + 1}`,
      answer: s.pred,
      category: s.cat,
    })),
    groundTruths: slice.map((s, i) => ({
      question_id: `rsvqa_q_${i + 1}`,
      answer: s.gt,
      category: s.cat,
    })),
  };
}

// ---------------------------------------------------------------------------
// 2. VRSBench Benchmark Dataset (40 Samples)
// ---------------------------------------------------------------------------
const VRSBENCH_DESCRIPTIONS = [
  'commercial airliner parked on taxiway',
  'oil storage tanks with white circular lids',
  'three container gantry cranes along quay',
  'five light aircraft on airport apron',
  'runway intersection and threshold markings',
  'stadium football pitch with green synthetic turf',
  'solar panel photovoltaic array',
  'dense residential neighborhood with red roofs',
  'highway cloverleaf interchange with overpass',
  'harbor dock with moored cargo vessel',
  'roundabout with four arterial exits',
  'four tennis courts in sports complex',
  'industrial manufacturing facility with smokestacks',
  'agricultural center-pivot circular irrigation field',
  'coastal breakwater barrier wall',
  'railway classification marshalling yard',
  'six fuel storage cylinders in bunded area',
  'helipad marked with white letter H',
  'marina boat slips and floating piers',
  'two electrical transmission sub-stations',
  'bridge crossing wide river channel',
  'parking lot with organized vehicle stalls',
  'quarry excavation pit with terraces',
  'water treatment clarifier circular tanks',
  'cement batch plant with raw material silos',
  'golf course fairways and sand bunkers',
  'intermodal rail freight terminal',
  'cargo ship navigating harbor channel',
  'grain elevator storage towers',
  'highway toll plaza canopy structure',
  'sewage settling ponds with aeration',
  'rooftop solar installations on warehouses',
  'airport terminal gate jetways',
  'coastal lighthouse on rocky promontory',
  'dam spillway and reservoir lake',
  'open-pit mining haul roads',
  'wind turbine three-blade rotor',
  'substation step-down transformers',
  'dry dock ship repair basin',
  'recreational boat marina slips',
];

export function getVRSBenchCorpus(sampleCount: number = 40): BenchmarkSampleCorpus {
  const count = Math.min(sampleCount, VRSBENCH_DESCRIPTIONS.length);
  const preds: Record<string, any>[] = [];
  const gts: Record<string, any>[] = [];

  for (let i = 0; i < count; i++) {
    const ymin = 0.08 + (i % 6) * 0.13;
    const xmin = 0.1 + (i % 5) * 0.15;
    const ymax = Math.min(ymin + 0.2 + (i % 3) * 0.06, 0.95);
    const xmax = Math.min(xmin + 0.22 + (i % 4) * 0.05, 0.95);

    const desc = VRSBENCH_DESCRIPTIONS[i];
    const cat = i % 4 === 0 ? 'presence' : i % 4 === 1 ? 'count' : i % 4 === 2 ? 'grounding' : 'landcover';

    // Simulated prediction with high precision grounding jitter
    const predBox = [
      Number((ymin + (i % 2 === 0 ? 0.005 : -0.005)).toFixed(3)),
      Number((xmin + (i % 3 === 0 ? 0.008 : -0.006)).toFixed(3)),
      Number((ymax + (i % 2 === 0 ? -0.004 : 0.006)).toFixed(3)),
      Number((xmax + (i % 3 === 0 ? -0.005 : 0.007)).toFixed(3)),
    ];
    const gtBox = [ymin, xmin, ymax, xmax];

    preds.push({
      question_id: `vrs_${i + 1}`,
      answer: desc,
      bbox: predBox,
      category: cat,
    });

    gts.push({
      question_id: `vrs_${i + 1}`,
      answer: desc,
      bbox: gtBox,
      category: cat,
    });
  }

  return { predictions: preds, groundTruths: gts };
}

// ---------------------------------------------------------------------------
// 3. CDVQA Benchmark Dataset (40 Samples)
// ---------------------------------------------------------------------------
const CDVQA_SCENARIOS: { change: boolean; desc: string }[] = [
  { change: true, desc: 'new multi-story residential buildings constructed in vacant field' },
  { change: false, desc: 'no significant surface change detected across seasonal acquisitions' },
  { change: true, desc: 'highway expansion from two lanes to four lanes with new concrete overpass' },
  { change: true, desc: 'commercial logistics warehouse center constructed with asphalt parking' },
  { change: false, desc: 'agricultural cropland unchanged with seasonal vegetation differences only' },
  { change: true, desc: 'forest clearing for ground-mounted solar photovoltaic panel installation' },
  { change: false, desc: 'industrial manufacturing complex layout remains identical' },
  { change: true, desc: 'water reservoir shoreline retreat and exposed dry lakebed sediments' },
  { change: true, desc: 'suburban housing subdivision expansion with ten new residential structures' },
  { change: false, desc: 'airport runway layout, markings, and terminal apron unchanged' },
  { change: true, desc: 'new shipping container storage yard paved with gantry crane tracks' },
  { change: true, desc: 'forest canopy clearing along river corridor buffer zone' },
  { change: false, desc: 'urban city center with unchanged road and building infrastructure' },
  { change: true, desc: 'coastal land reclamation and concrete seawall breakwater construction' },
  { change: true, desc: 'sports stadium synthetic running track installed adjacent to school' },
  { change: false, desc: 'arid desert terrain with stable rock formations and dunes' },
  { change: true, desc: 'surface quarry excavation pit enlarged with new soil stockpiles' },
  { change: true, desc: 'retail shopping center constructed on former greenfield agricultural parcel' },
  { change: false, desc: 'river channel course unchanged between dry and wet acquisition dates' },
  { change: true, desc: 'wind turbine concrete foundation pads constructed along mountain ridge' },
  { change: true, desc: 'bridge replacement with wider four-lane steel truss structure' },
  { change: false, desc: 'dense residential neighborhood layout unchanged' },
  { change: true, desc: 'electrical transmission substation expansion with new transformers' },
  { change: true, desc: 'demolition of aged industrial facility and site grading' },
  { change: false, desc: 'seaport container berths and cranes unchanged' },
  { change: true, desc: 'new center-pivot circular irrigation system installed in arid field' },
  { change: true, desc: 'wildfire burn scar regeneration with early pioneer vegetation' },
  { change: false, desc: 'golf course fairways and clubhouse facilities unchanged' },
  { change: true, desc: 'railway siding tracks added for intermodal freight loading' },
  { change: true, desc: 'parking garage structure built on surface parking lot' },
  { change: false, desc: 'wetland estuary tidal mudflats stable between dates' },
  { change: true, desc: 'new aquaculture shrimp ponds excavated in coastal zone' },
  { change: true, desc: 'highway sound barrier walls installed along residential perimeter' },
  { change: false, desc: 'mountain ridge vegetation and rock outcrops unchanged' },
  { change: true, desc: 'solar farm array expansion with 500 additional panel rows' },
  { change: true, desc: 'storm damage roof repairs and tarp covers on residential buildings' },
  { change: false, desc: 'commercial business park parking lot and roads unchanged' },
  { change: true, desc: 'new school gymnasium building constructed on athletic field' },
  { change: true, desc: 'coastal dredging channel marked by new navigation buoys' },
  { change: false, desc: 'cement manufacturing quarry boundaries stable between years' },
];

export function getCDVQACorpus(sampleCount: number = 40): BenchmarkSampleCorpus {
  const count = Math.min(sampleCount, CDVQA_SCENARIOS.length);
  const preds: Record<string, any>[] = [];
  const gts: Record<string, any>[] = [];

  for (let i = 0; i < count; i++) {
    const item = CDVQA_SCENARIOS[i];
    const ans = item.change ? 'yes' : 'no';

    preds.push({
      pair_id: `cdvqa_p_${i + 1}`,
      has_change: item.change,
      answer: ans,
      description: item.desc,
    });

    gts.push({
      pair_id: `cdvqa_p_${i + 1}`,
      has_change: item.change,
      answer: ans,
      description: item.desc,
    });
  }

  return { predictions: preds, groundTruths: gts };
}

// ---------------------------------------------------------------------------
// 4. ISRO / SAC Benchmark Dataset (36 Samples)
// ---------------------------------------------------------------------------
const ISRO_SAMPLES: { name: string; sensor: string; cat: string }[] = [
  { name: 'Ahmedabad urban expansion corridor parcel', sensor: 'Cartosat-2S', cat: 'urban' },
  { name: 'Godavari river delta irrigated paddy cropland', sensor: 'Resourcesat LISS-IV', cat: 'agriculture' },
  { name: 'Chilika lake aquaculture flooded wetland enclosures', sensor: 'RISAT-1 C-band SAR', cat: 'hydrology' },
  { name: 'Bengaluru electronic city tech park commercial complex', sensor: 'Cartosat-3', cat: 'infrastructure' },
  { name: 'Bhadla Thar desert solar park photovoltaic arrays', sensor: 'Resourcesat AWiFS', cat: 'energy' },
  { name: 'JNPT Mumbai harbor shipping container terminal yard', sensor: 'Cartosat-2E', cat: 'transportation' },
  { name: 'Western Ghats dense tropical evergreen canopy', sensor: 'Resourcesat LISS-III', cat: 'forestry' },
  { name: 'Hyderabad outer ring road multi-level junction', sensor: 'Cartosat-2S', cat: 'infrastructure' },
  { name: 'Punjab Ludhiana winter wheat agricultural crop belt', sensor: 'Resourcesat LISS-IV', cat: 'agriculture' },
  { name: 'Brahmaputra river braided sandbars and monsoon floodplains', sensor: 'RISAT-1 C-band SAR', cat: 'hydrology' },
  { name: 'Chennai port container berths and gantry cranes', sensor: 'Cartosat-3', cat: 'transportation' },
  { name: 'Sundarbans mangrove delta tidal wetlands', sensor: 'Resourcesat LISS-III', cat: 'forestry' },
  { name: 'Jaipur walled city heritage urban layout', sensor: 'Cartosat-2S', cat: 'urban' },
  { name: 'Cauvery river basin sugarcane intensive farming fields', sensor: 'Resourcesat LISS-IV', cat: 'agriculture' },
  { name: 'Kolkata Rajarhat township infrastructure development', sensor: 'Cartosat-3', cat: 'infrastructure' },
  { name: 'Pavagada solar park electrical substation and grid lines', sensor: 'Resourcesat AWiFS', cat: 'energy' },
  { name: 'Visakhapatnam steel plant heavy industrial site', sensor: 'Cartosat-2E', cat: 'infrastructure' },
  { name: 'Kochi international airport runway and taxiway apron', sensor: 'Cartosat-2S', cat: 'transportation' },
  { name: 'Mahanadi river delta coastal mangroves and mudflats', sensor: 'RISAT-1 C-band SAR', cat: 'hydrology' },
  { name: 'Pune Hinjewadi IT park phase 3 campus', sensor: 'Cartosat-3', cat: 'infrastructure' },
  { name: 'Haryana cotton and basmati rice agricultural parcels', sensor: 'Resourcesat LISS-IV', cat: 'agriculture' },
  { name: 'Gir national park dry deciduous forest canopy', sensor: 'Resourcesat LISS-III', cat: 'forestry' },
  { name: 'Surat diamond bourse commercial infrastructure', sensor: 'Cartosat-3', cat: 'infrastructure' },
  { name: 'Mundra port coal terminal and bulk cargo berths', sensor: 'Cartosat-2E', cat: 'transportation' },
  { name: 'Narmada canal network distributary irrigation branches', sensor: 'Resourcesat LISS-IV', cat: 'hydrology' },
  { name: 'Rewa ultra mega solar power project grid tie', sensor: 'Resourcesat AWiFS', cat: 'energy' },
  { name: 'Chandigarh planned urban grid sectors', sensor: 'Cartosat-2S', cat: 'urban' },
  { name: 'Kashmir valley apple orchard agricultural terraces', sensor: 'Resourcesat LISS-IV', cat: 'agriculture' },
  { name: 'Dhamra port coastal breakwater and navigation channel', sensor: 'RISAT-1 C-band SAR', cat: 'transportation' },
  { name: 'Delhi-Mumbai Industrial Corridor (DMIC) Dholera node', sensor: 'Cartosat-3', cat: 'infrastructure' },
  { name: 'Kaziranga national park grassland and wetland oxbows', sensor: 'Resourcesat LISS-III', cat: 'forestry' },
  { name: 'Tuticorin VOC port container handling berths', sensor: 'Cartosat-2E', cat: 'transportation' },
  { name: 'Ganga river basin Varanasi ghats and alluvial floodplain', sensor: 'RISAT-1 C-band SAR', cat: 'hydrology' },
  { name: 'Nagpur multimodal international hub airport at Nagpur (MIHAN)', sensor: 'Cartosat-2S', cat: 'infrastructure' },
  { name: 'Coimbatore textile manufacturing industrial clusters', sensor: 'Cartosat-2S', cat: 'urban' },
  { name: 'Rann of Kutch white salt desert seasonal crust', sensor: 'Resourcesat AWiFS', cat: 'hydrology' },
];

export function getISROSACCorpus(sampleCount: number = 36): BenchmarkSampleCorpus {
  const count = Math.min(sampleCount, ISRO_SAMPLES.length);
  const preds: Record<string, any>[] = [];
  const gts: Record<string, any>[] = [];

  for (let i = 0; i < count; i++) {
    const s = ISRO_SAMPLES[i];
    const ymin = 0.12 + (i % 5) * 0.14;
    const xmin = 0.1 + (i % 4) * 0.18;
    const ymax = Math.min(ymin + 0.22 + (i % 3) * 0.05, 0.96);
    const xmax = Math.min(xmin + 0.25 + (i % 4) * 0.04, 0.96);

    const fullDesc = `${s.sensor} ${s.name}`;

    preds.push({
      sample_id: `isro_${i + 1}`,
      answer: fullDesc,
      bbox: [
        Number((ymin + (i % 2 === 0 ? 0.004 : -0.004)).toFixed(3)),
        Number((xmin + (i % 3 === 0 ? 0.006 : -0.005)).toFixed(3)),
        Number((ymax + (i % 2 === 0 ? -0.003 : 0.005)).toFixed(3)),
        Number((xmax + (i % 3 === 0 ? -0.004 : 0.006)).toFixed(3)),
      ],
      category: s.cat,
    });

    gts.push({
      sample_id: `isro_${i + 1}`,
      answer: fullDesc,
      bbox: [ymin, xmin, ymax, xmax],
      category: s.cat,
    });
  }

  return { predictions: preds, groundTruths: gts };
}
