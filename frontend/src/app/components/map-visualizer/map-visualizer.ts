import { Component, ElementRef, OnInit, ViewChild, effect, input } from '@angular/core';
import * as L from 'leaflet';
import { MeteoData } from '../../models/weather.interface';

@Component({
  selector: 'app-map-visualizer',
  standalone: true,
  templateUrl: './map-visualizer.html'
})
export class MapVisualizerComponent implements OnInit {
  @ViewChild('mapContainer', { static: true }) mapContainer!: ElementRef;
  weatherData = input<MeteoData[]>([]);

  private map!: L.Map;

  // DEMO LOCATIONS: We map specific risks to specific cities for the demo
  private readonly LOCATIONS: Record<string, [number, number]> = {
    'BENI_MELLAL': [32.33725, -6.34983], // Hot Zone
    'TANGIER': [35.7595, -5.8340],       // Windy Zone
    'KHOURIBGA': [32.8807, -6.9063]      // Default / HQ
  };

  constructor() {
    // React to data changes
    effect(() => {
      const data = this.weatherData();
      if (this.map && data.length > 0) this.updateMarkers(data);
    });
  }

  ngOnInit() {
    this.initMap();
  }

  private initMap(): void {
    const streetLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap'
    });

    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles © Esri'
    });

    this.map = L.map(this.mapContainer.nativeElement, {
      center: [32.5, -6.0], // Zoomed out slightly to see both Tangier and Beni Mellal
      zoom: 6,
      layers: [streetLayer]
    });

    L.control.layers({ "Street": streetLayer, "Satellite": satelliteLayer }).addTo(this.map);
  }

  private updateMarkers(data: MeteoData[]): void {
    // 1. Clear existing markers
    this.map.eachLayer((layer) => {
        if (layer instanceof L.CircleMarker) this.map.removeLayer(layer);
    });

    // 2. FIND ALL ALERTS (Not just the worst one)
    // This allows us to show a Heatwave in Beni Mellal AND a Storm in Tangier simultaneously if data permits
    const activeAlerts = data.filter(d => d.alert !== null);

    if (activeAlerts.length === 0) return;

    // 3. PLOT MARKERS BASED ON TYPE
    activeAlerts.forEach(day => {
        if (!day.alert) return;

        let coords = this.LOCATIONS['KHOURIBGA']; // Default
        let locationName = "Khouribga";

        // LOGIC: Map Disaster Type -> Location
        if (day.alert.type.includes('HEAT') || day.alert.type.includes('DRY')) {
            coords = this.LOCATIONS['BENI_MELLAL'];
            locationName = "Beni Mellal";
        } else if (day.alert.type.includes('WIND') || day.alert.type.includes('STORM')) {
            coords = this.LOCATIONS['TANGIER'];
            locationName = "Tangier";
        }

        const color = day.alert.level === 'CRITICAL' ? '#ef4444' : '#f97316';

        // Check if a marker already exists at these coords to avoid stacking (optional optimization)
        // For simplicity, we just plot it.
        
        L.circleMarker(coords, {
            radius: 14,
            fillColor: color,
            color: '#fff',
            weight: 3,
            opacity: 1,
            fillOpacity: 0.9
        })
        .bindPopup(`
            <div style="text-align:center">
            <strong style="color:${color}">${day.alert.type}</strong><br>
            <span style="font-size:11px; color:#666">${locationName} Station</span><br>
            <b>${day.date}</b><br>
            ${day.alert.message}
            </div>
        `)
        .addTo(this.map);
    });
  }
}