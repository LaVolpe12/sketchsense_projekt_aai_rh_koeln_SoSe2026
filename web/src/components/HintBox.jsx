/**
 * HintBox
 * -------
 * Shows category-specific drawing tips when the model's top confidence
 * is below a threshold, helping the user improve their sketch.
 *
 * Props:
 *   topLabel: string       current top prediction
 *   confidence: number     0–1
 */

import React from "react";

// Hints per category — what details make this drawing recognisable?
const HINTS = {
  ant:              ["Draw a small oval body", "Add 6 thin legs", "Add antennae on the head"],
  bear:             ["Draw a large round body", "Add small round ears", "Draw a snout protruding forward"],
  butterfly:        ["Draw two large wings on each side", "Add a thin body in the center", "Add antennae curving upward"],
  camel:            ["Draw one or two humps on the back", "Add long thin legs", "Draw a long neck and small head"],
  cat:              ["Add pointy triangular ears", "Draw whiskers on the face", "Add a long curved tail"],
  dog:              ["Add floppy or perky ears", "Draw a snout and nose", "Add a wagging tail"],
  dolphin:          ["Draw a curved fish-like body", "Add a dorsal fin on top", "Draw a horizontal tail fluke"],
  elephant:         ["Draw a large round body", "Add a long curved trunk", "Draw big ears on the sides"],
  frog:             ["Draw a round body low to ground", "Add large bulging eyes on top", "Draw webbed feet spread out"],
  giraffe:          ["Draw a very long neck", "Add small horn-like ossicones", "Draw a spotted square body"],
  horse:            ["Draw an elongated body", "Add four long straight legs", "Draw a flowing mane on the neck"],
  kangaroo:         ["Draw a large hind body", "Add a long thick tail for balance", "Draw a small pouch on the belly"],
  lion:             ["Draw a large fluffy mane around the head", "Add a round face inside", "Draw a tufted tail"],
  monkey:           ["Draw a round face with big ears", "Add a curved tail", "Draw long arms reaching down"],
  octopus:          ["Draw a round head/body", "Add 8 wavy tentacles below", "Add two large eyes"],
  penguin:          ["Draw a round black body", "Add a white oval belly", "Draw small flippers on the sides"],
  rabbit:           ["Draw very long upright ears", "Add a round fluffy body", "Draw a small round tail"],
  shark:            ["Draw a pointed nose and streamlined body", "Add a large triangular dorsal fin", "Draw a crescent tail fin"],
  snake:            ["Draw a long S-shaped wavy body", "Add a small triangular head", "Draw a forked tongue"],
  spider:           ["Draw a round body with a smaller head", "Add 8 thin legs", "Add small eyes on the head"],
  whale:            ["Draw a very large rounded body", "Add a horizontal tail fluke", "Draw a blowhole spout on top"],
  zebra:            ["Draw a horse-like body", "Add vertical black stripes all over", "Draw a striped mane"],
  airplane:         ["Add wings on both sides", "Draw a tail fin", "Add a nose cone at the front"],
  ambulance:        ["Draw a boxy vehicle shape", "Add a red cross on the side", "Draw a light bar on top"],
  bicycle:          ["Draw two circles for wheels", "Connect with a triangular frame", "Add a handlebar and seat"],
  bus:              ["Draw a long rectangular body", "Add many windows in a row", "Draw large wheels underneath"],
  car:              ["Add round wheels", "Draw a windshield", "Add side windows and doors"],
  helicopter:       ["Draw a long body", "Add large rotor blades on top", "Draw a small tail rotor"],
  motorbike:        ["Draw two wheels close together", "Add a low seat and handlebars", "Draw an engine block in the middle"],
  sailboat:         ["Draw a pointed hull at the bottom", "Add a tall vertical mast", "Draw a triangular sail"],
  train:            ["Draw several rectangular carriages in a row", "Add wheels along the bottom", "Draw a smokestack on the engine"],
  truck:            ["Draw a large rectangular cargo section", "Add a cab at the front", "Draw big double wheels"],
  cactus:           ["Draw a tall vertical column", "Add horizontal arm branches", "Add small spines along the edges"],
  campfire:         ["Draw a triangular flame shape", "Add logs crossing underneath", "Draw smaller inner flames"],
  cloud:            ["Draw several overlapping bumpy circles", "Make the bottom flat", "Add more bumps on top"],
  flower:           ["Draw petals in a circle around the center", "Add a circular center", "Draw a stem and leaves"],
  lightning:        ["Draw a zigzag bolt shape", "Make it wide at the top and pointed at the bottom", "Add a glow effect around it"],
  moon:             ["Draw a crescent shape", "Make one side curved and one side concave", "Add a few stars nearby"],
  mountain:         ["Draw a large triangle", "Add a snow cap at the peak", "Draw smaller mountains behind"],
  mushroom:         ["Draw a wide dome-shaped cap", "Add a thick short stem below", "Draw spots on the cap"],
  rainbow:          ["Draw several concentric arcs", "Use bands of color from red to violet", "Add clouds at each end"],
  snowflake:        ["Draw six symmetric arms from the center", "Add small branches on each arm", "Make it perfectly symmetric"],
  sun:              ["Draw a circle in the center", "Add rays pointing outward all around", "Make the rays evenly spaced"],
  tree:             ["Draw a triangle or cloud shape for the crown", "Add a thick rectangular trunk", "Draw some branches"],
  apple:            ["Draw a round shape with a dip at the top", "Add a small stem", "Draw a leaf beside the stem"],
  banana:           ["Make a long curved crescent shape", "Add a tip at each end", "Draw the ridge lines along the curve"],
  broccoli:         ["Draw a bumpy cloud shape on top", "Add a thick trunk stem", "Draw branches connecting top and stem"],
  cake:             ["Draw a round or rectangular layer base", "Add a second layer on top", "Draw candles or frosting drips"],
  "coffee cup":     ["Draw a cylindrical cup", "Add a handle on the side", "Draw steam rising from the top"],
  donut:            ["Draw a circle with a hole in the middle", "Add icing dripping over the top", "Draw sprinkles on the icing"],
  hamburger:        ["Draw a round top bun", "Add layers of patty and lettuce", "Draw a round bottom bun"],
  "hot dog":        ["Draw a long oval sausage shape", "Add a bun around it", "Draw mustard lines on top"],
  pineapple:        ["Draw an oval body", "Add a spiky leafy crown on top", "Draw a diamond grid pattern on the body"],
  pizza:            ["Draw a triangle slice", "Add a crust edge at the top", "Draw toppings like circles and lines"],
  strawberry:       ["Draw a heart-shaped body", "Add a leafy top", "Draw small seeds as dots on the surface"],
  watermelon:       ["Draw a large semicircle", "Add a green outer rind", "Draw black seeds inside the red flesh"],
  backpack:         ["Draw a rectangle with rounded corners", "Add two shoulder straps", "Draw a front pocket"],
  bed:              ["Draw a rectangular mattress", "Add a headboard at one end", "Draw a pillow on top"],
  book:             ["Draw a rectangle", "Add a spine on one side", "Draw horizontal lines for pages"],
  chair:            ["Draw a flat seat", "Add four legs underneath", "Draw a backrest rising from the back"],
  clock:            ["Draw a circle", "Add clock hands pointing to a time", "Draw hour markers around the edge"],
  cup:              ["Draw a cylinder shape", "Add a handle on the side", "Make the top wider than the base"],
  envelope:         ["Draw a rectangle", "Add a V-shaped flap at the top", "Draw diagonal lines meeting in the center"],
  eyeglasses:       ["Draw two circles side by side", "Connect them with a small bridge", "Add arms extending to the sides"],
  hammer:           ["Draw a long handle", "Add a rectangular head on top perpendicular to the handle", "Make the head thick and heavy-looking"],
  key:              ["Draw a round head with a hole", "Add a long thin shaft", "Draw teeth along the bottom of the shaft"],
  pencil:           ["Draw a long thin rectangle", "Add a pointed tip at one end", "Draw a small eraser at the other end"],
  scissors:         ["Draw two blades crossing in an X", "Add circular finger holes at the top", "Make the blades narrow and pointed"],
  shoe:             ["Draw a sole curving up at the front", "Add a heel at the back", "Draw the upper part of the shoe above the sole"],
  telephone:        ["Draw a rectangular base", "Add a curved handset on top", "Draw buttons in the middle"],
  umbrella:         ["Draw a dome-shaped top", "Add a curved handle at the bottom", "Draw ribs along the inside of the dome"],
  barn:             ["Draw a large rectangular body", "Add a triangular or gambrel roof", "Draw an X on the large door"],
  bridge:           ["Draw two towers rising from the sides", "Add cables or arches connecting them", "Draw the road deck along the bottom"],
  castle:           ["Draw several tall towers with battlements", "Add a gate in the center", "Draw arrow slits in the walls"],
  church:           ["Draw a rectangular building", "Add a tall pointed steeple", "Draw a cross at the top"],
  house:            ["Draw a square body", "Add a triangular roof on top", "Draw a door and windows"],
  lighthouse:       ["Draw a tall narrow tower", "Add a glowing light room at the top", "Draw diagonal stripes on the tower"],
  skyscraper:       ["Draw a very tall narrow rectangle", "Add many rows of small windows", "Make it taper slightly toward the top"],
  tent:             ["Draw a large triangle", "Add a small entrance at the bottom center", "Draw guy ropes extending outward"],
  cello:            ["Draw a figure-8 shaped body", "Add a long thin neck", "Draw strings running from the bottom to the tuning pegs"],
  drums:            ["Draw a large cylinder for the bass drum", "Add a snare drum to the side", "Draw sticks crossed above"],
  guitar:           ["Draw a figure-8 shaped body", "Add a long neck with frets", "Draw strings from the bridge to the head"],
  harp:             ["Draw a tall triangular frame", "Add strings running diagonally inside", "Draw a curved neck at the top"],
  piano:            ["Draw a long rectangular keyboard", "Add black keys among the white ones", "Draw the body of the piano above"],
  saxophone:        ["Draw a curved conical tube", "Add keys along the body", "Draw a flared bell at the bottom"],
  trumpet:          ["Draw a long tube that curves around", "Add three valves in the middle", "Draw a flared bell at the end"],
  violin:           ["Draw a figure-8 shaped body", "Add a long thin neck", "Draw an S-shaped f-hole on the body"],
  "baseball bat":   ["Draw a long cylinder that widens at one end", "Make the handle thin", "Draw a round knob at the grip end"],
  basketball:       ["Draw a circle", "Add curved lines dividing it into panels", "Make it orange"],
  "hockey stick":   ["Draw a long straight handle", "Add a flat blade angled at the bottom", "Draw the curve where handle meets blade"],
  skateboard:       ["Draw a long oval deck", "Add two axles underneath", "Draw four small wheels"],
  "soccer ball":    ["Draw a circle", "Add pentagons and hexagons in a pattern", "Make some patches dark"],
  yoga:             ["Draw a stick figure in a seated pose", "Add legs crossed in lotus position", "Draw arms resting on the knees"],
  crown:            ["Draw a band across the middle", "Add pointed spikes rising from the top", "Draw jewels on the band"],
  diamond:          ["Draw a triangle pointing down", "Add a flat top edge", "Draw facet lines inside"],
  hourglass:        ["Draw two triangles pointing toward each other", "Connect them in the middle", "Add a frame around the outside"],
  "hot air balloon":["Draw a large teardrop or oval balloon", "Add vertical stripes or panels", "Draw a small basket hanging below"],
  "smiley face":    ["Draw a circle", "Add two dots for eyes", "Draw a curved smile"],
  snowman:          ["Draw three circles stacked from large to small", "Add a hat on top", "Draw buttons and a carrot nose"],
  star:             ["Draw five points evenly spaced", "Connect them with lines", "Make all points the same length"],
};

const THRESHOLD = 0.45;

export default function HintBox({ topLabel, confidence }) {
  if (!topLabel || confidence >= THRESHOLD) return null;

  const hints = HINTS[topLabel] ?? [];
  if (hints.length === 0) return null;

  return (
    <div className="hint-box">
      <div className="hint-header">
        <span className="hint-icon">💡</span>
        <span>Zeichentipps für: <strong>{topLabel}</strong></span>
      </div>
      <ul className="hint-list">
        {hints.map((h) => (
          <li key={h}>{h}</li>
        ))}
      </ul>
    </div>
  );
}
