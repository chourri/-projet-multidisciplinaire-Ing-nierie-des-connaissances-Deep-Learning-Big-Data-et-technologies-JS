import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderComponent } from '../../components/header/header';

interface MapImage {
  id: number;
  code: string; // Added for the "Archive" feel
  url: string;
  title: string;
  source: string;
  type: 'Satellite' | 'Disaster' | 'Analysis';
  date: string;
}

@Component({
  selector: 'app-maps-page',
  standalone: true,
  imports: [CommonModule, HeaderComponent],
  templateUrl: './maps-page.html'
})
export class MapsPageComponent {
  
  selectedImage = signal<MapImage | null>(null);

  // Added 'code' property for that official look
  galleryImages: MapImage[] = [
    { 
      id: 1, 
      code: 'SAT-OPT-024',
      url: 'assets/satellite/sat_morocco.jpg', 
      title: 'Regional Topography & Cloud Cover', 
      source: 'Sentinel-2A', 
      type: 'Satellite', 
      date: '2026-01-15' 
    },
    { 
      id: 2, 
      code: 'SAR-FLD-102',
      url: 'assets/satellite/disaster_flood.jpg', 
      title: 'Gharb Plain Flood Extent', 
      source: 'Sentinel-1 SAR', 
      type: 'Disaster', 
      date: '2026-01-28' 
    },
    { 
      id: 3, 
      code: 'TRM-LST-88',
      url: 'assets/satellite/sat_heat.jpg', 
      title: 'Surface Temperature Anomalies', 
      source: 'Landsat 8 TIRS', 
      type: 'Analysis', 
      date: '2026-02-01' 
    },
    { 
      id: 4, 
      code: 'VEG-MOD-12',
      url: 'assets/satellite/sat_ndvi.jpg', 
      title: 'Vegetation Health Index (NDVI)', 
      source: 'MODIS Terra', 
      type: 'Analysis', 
      date: '2026-01-10' 
    },
    { 
      id: 5, 
      code: 'MET-STM-05',
      url: 'assets/satellite/sat_storm.jpg', 
      title: 'Atlantic Storm Front', 
      source: 'Meteosat-11', 
      type: 'Satellite', 
      date: '2026-02-03' 
    },
    { 
      id: 6, 
      code: 'HYD-RES-09',
      url: 'assets/satellite/disaster_drought.jpg', 
      title: 'Reservoir Depletion Levels', 
      source: 'SPOT-7', 
      type: 'Disaster', 
      date: '2026-01-22' 
    }
  ];

  openLightbox(image: MapImage) {
    this.selectedImage.set(image);
    document.body.style.overflow = 'hidden';
  }

  closeLightbox() {
    this.selectedImage.set(null);
    document.body.style.overflow = 'auto';
  }
}