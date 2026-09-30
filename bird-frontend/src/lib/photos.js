/**
 * The photographs the page is built around, with their credits.
 *
 * All are openly licensed on Wikimedia Commons, resized for the web and served
 * from public/photos so the app still runs with no network. Each entry comes in
 * two sizes: `src` for full-bleed use and `thumb` (800px) for tiles.
 *
 * `position` is the CSS object-position that keeps the bird in frame when the
 * photo is cropped to a shape it was not shot for.
 */

const photo = (file, fields) => ({
  src: `/photos/${file}.jpg`,
  thumb: `/photos/${file}-sm.jpg`,
  ...fields,
})

export const PHOTOS = {
  macaw: photo('macaw-blue-gold', {
    name: 'Blue-and-yellow macaw',
    sci: 'Ara ararauna',
    alt: 'Close portrait of a blue-and-yellow macaw against a warm orange background',
    position: '22% 40%',
    author: 'H. Zell',
    license: 'CC BY-SA 3.0',
    licenseUrl: 'https://creativecommons.org/licenses/by-sa/3.0',
    source: 'https://commons.wikimedia.org/wiki/File:Ara_ararauna_01.jpg',
  }),
  kingfisher: photo('kingfisher', {
    name: 'Common kingfisher',
    sci: 'Alcedo atthis',
    alt: 'A common kingfisher perched on a rock, facing right',
    position: '38% 50%',
    author: '谷崎かおる',
    license: 'Public domain',
    licenseUrl: null,
    source:
      'https://commons.wikimedia.org/wiki/File:Common_kingfisher_in_Japan,_January_2019_-_%E8%B0%B7%E5%B4%8E7303.jpg',
  }),
  owl: photo('barn-owl', {
    name: 'Barn owl',
    sci: 'Tyto alba',
    alt: 'A barn owl looking toward the camera against dark foliage',
    position: '50% 28%',
    author: 'Michael Gäbler',
    license: 'CC BY 3.0',
    licenseUrl: 'https://creativecommons.org/licenses/by/3.0',
    source: 'https://commons.wikimedia.org/wiki/File:Tyto_alba_(Scopoli,_1769).jpg',
  }),
  scarlet: photo('scarlet-macaws', {
    name: 'Scarlet macaws',
    sci: 'Ara macao',
    alt: 'Two scarlet macaws standing on a rock at the mouth of a cave',
    position: '72% 60%',
    author: 'Charles J. Sharp',
    license: 'CC BY-SA 4.0',
    licenseUrl: 'https://creativecommons.org/licenses/by-sa/4.0',
    source: 'https://commons.wikimedia.org/wiki/File:Scarlet_macaws_(Ara_macao_macao)_pair_Yasuni.jpg',
  }),
  wren: photo('wren-singing', {
    name: 'Eurasian wren, singing',
    sci: 'Troglodytes troglodytes',
    alt: 'A Eurasian wren singing with its beak open, tail cocked',
    position: '62% 45%',
    author: 'Alexis Lours',
    license: 'CC BY 4.0',
    licenseUrl: 'https://creativecommons.org/licenses/by/4.0',
    source: 'https://commons.wikimedia.org/wiki/File:Eurasian_wren_2023_12_31_02.jpg',
  }),
  brilliant: photo('brilliant-hummingbird', {
    name: 'Fawn-breasted brilliant',
    sci: 'Heliodoxa rubinoides',
    alt: 'A fawn-breasted brilliant hummingbird perched on a mossy branch',
    position: '55% 40%',
    author: 'Andy Morffew',
    license: 'CC BY 2.0',
    licenseUrl: 'https://creativecommons.org/licenses/by/2.0',
    source:
      'https://commons.wikimedia.org/wiki/File:Fawn-breasted_Brilliant_hummingbird_in_Ecuador_-_(54564276957).jpg',
  }),
  rufous: photo('rufous-tailed-hummingbird', {
    name: 'Rufous-tailed hummingbird',
    sci: 'Amazilia tzacatl',
    alt: 'A rufous-tailed hummingbird hovering at yellow flowers',
    position: '50% 45%',
    author: 'Andy Morffew',
    license: 'CC BY 2.0',
    licenseUrl: 'https://creativecommons.org/licenses/by/2.0',
    source: 'https://commons.wikimedia.org/wiki/File:Rufous-tailed_Hummingbird_(54567044907).jpg',
  }),
  egret: photo('great-egret', {
    name: 'Great egret',
    sci: 'Ardea alba',
    alt: 'A great egret in flight against a grey sky',
    position: '45% 45%',
    author: 'Hobbyfotowiki',
    license: 'CC0',
    licenseUrl: 'https://creativecommons.org/publicdomain/zero/1.0/',
    source:
      'https://commons.wikimedia.org/wiki/File:Great_egret_(great_egret_(Ardea_alba)_at_Federsee,_Germany)_at_Federsee.jpg',
  }),
}

/** The hero cycles through these, in order. */
export const HERO_SLIDES = [PHOTOS.macaw, PHOTOS.kingfisher, PHOTOS.owl]
