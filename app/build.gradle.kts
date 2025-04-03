plugins {
    alias(libs.plugins.android.application)
+   // If you plan to use Kotlin (recommended), add this:
+   // alias(libs.plugins.kotlinAndroid)
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
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
+   // If using Kotlin:
+   // kotlinOptions {
+   //    jvmTarget = "11"
+   // }
+
+   // Add this block to prevent compression of TFLite files
+   androidResources {
+       noCompress.add(".tflite")
+       noCompress.add(".bin") // Also prevent compression for vocab/filter files
+   }
+
+   // If using viewBinding (recommended for UI interaction)
+   // buildFeatures {
+   //     viewBinding = true
+   // }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
+   implementation(libs.tensorflow.lite)
+   // implementation(libs.tensorflow.lite.support) // Add if you use support library features

    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}