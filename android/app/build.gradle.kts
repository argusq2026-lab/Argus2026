plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.argus.edge"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.argus.edge"
        // Snapdragon 8 Elite (SM8750) ships Android 15. 31 is a safe floor that
        // still covers the CameraX APIs used here.
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // arm64-v8a only. The Hexagon NPU exists on no other ABI, and shipping
        // a 32-bit slice would produce a build that installs and then cannot
        // load the QNN backend -- exactly the quiet degradation this project
        // refuses everywhere else.
        ndk { abiFilters += "arm64-v8a" }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    sourceSets {
        getByName("main").java.srcDirs("src/main/kotlin")
        getByName("test").java.srcDirs("src/test/kotlin")
    }

    // The cross-language wire fixture lives with the Python tests that generate
    // it. Putting it on the unit-test resource path means both platforms check
    // themselves against one file rather than two drifting copies.
    testOptions {
        unitTests.all { it.systemProperty("argus.fixtures", "${rootDir.parentFile}/tests/data") }
    }

    packaging {
        // The QNN backend .so set from the QAIRT SDK is dropped in here by
        // whoever has Qualcomm portal access; see android/README.md. It is
        // deliberately not vendored into the repo.
        jniLibs.pickFirsts += "**/libQnn*.so"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // ONNX Runtime with the QNN execution provider compiled in. The Qualcomm
    // backend libraries it dlopens at runtime are NOT in this artifact -- see
    // QnnDetector for what happens when they are absent.
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.22.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}
