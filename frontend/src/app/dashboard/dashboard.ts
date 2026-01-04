import { Component, inject, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';
import { WeatherService } from '../services/weather';
import { HeaderComponent } from '../components/header/header';
import { MapVisualizerComponent } from '../components/map-visualizer/map-visualizer';
import { AlertListComponent } from '../components/alert-list/alert-list';
import { ImageSliderComponent } from '../components/image-slider/image-slider';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, HeaderComponent, MapVisualizerComponent, AlertListComponent, ImageSliderComponent],
  templateUrl: './dashboard.html'
})
export class DashboardComponent {
  private weatherService = inject(WeatherService);
  forecastSignal = toSignal(this.weatherService.getForecast(), { initialValue: null });

  weatherData = computed(() => {
    const response = this.forecastSignal();
    return response ? response.data : [];
  });

  criticalAlert = computed(() => {
    const data = this.weatherData();
    return data.find(day => day.alert !== null)?.alert || null;
  });
}