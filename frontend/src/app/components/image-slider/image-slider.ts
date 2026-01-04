import { Component, OnDestroy, OnInit, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-image-slider',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './image-slider.html'
})
export class ImageSliderComponent implements OnInit, OnDestroy {
  
  images = signal([
    { title: 'Flood Map: Gharb Region', url: 'https://preview.redd.it/un1cm7q348w71.jpg?width=640&crop=smart&auto=webp&s=2b4131a2d49e5ae5a2ebb4429ba88cd5499285f0', date: '2026-01-28' },
    { title: 'Heatwave Index: Marrakesh', url: 'https://upload.wikimedia.org/wikipedia/commons/1/1a/Satellite_image_of_Morocco_in_January_2002.jpg', date: '2026-02-01' },
    { title: 'NDVI Vegetation Health', url: 'https://www.moroccoworldnews.com/wp-content/uploads/2024/11/nasa-satellite-images-capture-droughts-impact-on-morocco.jpeg', date: '2026-01-15' },
    { title: 'Soil Moisture Anomalies', url: 'https://www.mediastorehouse.com/t/617/morocco-neighbouring-countries-37177129.jpg.webp', date: '2026-01-20' },
    { title: 'Atmospheric Pressure', url: 'https://as2.ftcdn.net/jpg/05/53/70/75/1000_F_553707546_M44rAj3z9IetFCCP8OMpi9QqMOujJl6z.jpg', date: '2026-02-02' }
  ]);

  currentIndex = signal(0);
  private intervalId: any; // Variable to hold the timer

  // Computed signal to always show 3 images (wrapping around)
  visibleImages = computed(() => {
    const start = this.currentIndex();
    const allImages = this.images();
    
    return [
      allImages[start % allImages.length],
      allImages[(start + 1) % allImages.length],
      allImages[(start + 2) % allImages.length]
    ];
  });

  // 1. START the timer when component loads
  ngOnInit() {
    this.startAutoSlide();
  }

  // 2. CLEANUP the timer when component is destroyed (prevents memory leaks)
  ngOnDestroy() {
    this.stopAutoSlide();
  }

  next() {
    this.currentIndex.update(i => (i + 1) % this.images().length);
  }

  prev() {
    this.currentIndex.update(i => (i - 1 + this.images().length) % this.images().length);
  }

  // Helper to start the interval (5 seconds)
  startAutoSlide() {
    // Only start if not already running
    if (!this.intervalId) {
      this.intervalId = setInterval(() => {
        this.next();
      }, 5000); // <--- 5000ms = 5 Seconds
    }
  }

  // Helper to stop the interval
  stopAutoSlide() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
}