// Inside app/build.gradle.kts

plugins {
    alias(libs.plugins.android.application)
    // If you plan to use Kotlin (recommended), add this:
    // alias(libs.plugins.kotlinAndroid)
}

android {
    namespace = "com.example.projectvoice"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.projectvoice"
        minSdk = 34 // Whisper might require lower, but keeping yours for now
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        // JTransforms might require Java 8+ compatibility
        sourceCompatibility = JavaVersion.VERSION_1_8 // Changed from 11 to 8 for wider compatibility if needed, but 11 should be fine
        targetCompatibility = JavaVersion.VERSION_1_8 // Changed from 11 to 8
    }
    // If using Kotlin:
    // kotlinOptions {
    //    jvmTarget = "1.8" // Ensure Kotlin JVM target matches
    // }

    // Add this block to prevent compression of TFLite files
    androidResources {
        noCompress.add(".tflite")
        noCompress.add(".bin") // Also prevent compression for vocab/filter files
    }

    // If using viewBinding (recommended for UI interaction)
    // buildFeatures {
    //     viewBinding = true
    // }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.tensorflow.lite)
    // implementation(libs.tensorflow.lite.support) // Add if you use support library features

    // --- Add JTransforms dependency ---
    implementation("com.github.wendykierp:JTransforms:3.1") {
        // Exclude conflicting dependencies if they arise (unlikely here)
        // exclude(group = "junit", module="junit")
    }

    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}