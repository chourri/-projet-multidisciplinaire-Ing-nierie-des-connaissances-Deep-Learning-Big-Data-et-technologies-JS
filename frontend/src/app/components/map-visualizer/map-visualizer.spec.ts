import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MapVisualizer } from './map-visualizer';

describe('MapVisualizer', () => {
  let component: MapVisualizer;
  let fixture: ComponentFixture<MapVisualizer>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MapVisualizer]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MapVisualizer);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
