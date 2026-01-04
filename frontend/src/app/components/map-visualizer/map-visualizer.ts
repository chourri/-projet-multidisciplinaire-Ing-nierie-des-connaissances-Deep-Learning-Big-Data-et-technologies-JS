import { Component, ElementRef, OnInit, ViewChild, Input, effect, input } from '@angular/core';
import * as L from 'leaflet';
import { MeteoData } from '../../models/weather.interface';

@Component({
  selector: 'app-map-visualizer',
  standalone: true,
  templateUrl: './map-visualizer.html'
})
export class MapVisualizerComponent implements OnInit {
  @ViewChild('mapContainer', { static: true }) mapContainer!: ElementRef;
  
  // Input Signal (Angular 17+)
  weatherData = input<MeteoData[]>([]);

  private map!: L.Map;

  constructor() {
    // React to data changes
    effect(() => {
      const data = this.weatherData();
      if (this.map && data.length > 0) {
        this.updateMarkers(data);
      }
    });
  }

  ngOnInit() {
    this.initMap();
  }

  private initMap(): void {
    // Center on Morocco [Lat, Lng]
    this.map = L.map(this.mapContainer.nativeElement).setView([31.7917, -7.0926], 5);

    // Free OpenStreetMap Tiles (GDACS uses similar free layers)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(this.map);
  }

  private updateMarkers(data: MeteoData[]): void {
    // Clear existing markers (logic simplified for demo)
    this.map.eachLayer((layer) => {
        if (layer instanceof L.Marker) {
            this.map.removeLayer(layer);
        }
    });

    data.forEach(day => {
      if (day.alert) {
        // We simulate a location for the alert since your data doesn't have Lat/Lng yet.
        // In real app, MeteoData should have coordinates.
        // For now, we plot them around Marrakesh/Casablanca randomly for demo.
        const lat = 31.62 + (Math.random() * 2 - 1); 
        const lng = -7.98 + (Math.random() * 2 - 1);

        const color = day.alert.level === 'CRITICAL' ? 'red' : 'orange';
        
        // Custom Circle Marker
        L.circleMarker([lat, lng], {
          radius: 10,
          fillColor: color,
          color: '#fff',
          weight: 1,
          opacity: 1,
          fillOpacity: 0.8
        })
        .bindPopup(`<b>${day.alert.type}</b><br>${day.date}<br>${day.alert.message}`)
        .addTo(this.map);
      }
    });
  }
}